"""Facet extraction at document intake (Pillar 4, hook 1).

A dedicated, small, FLAT Qwen JSON call — deliberately not folded into the
big case-memory extraction schema (Gemma-hang + truncation lessons: local
models get small flat schemas). Runs once per case when new documents are
reflected and the case has no matchable facets yet; the backfill CLI reuses
the same call for existing cases. Advisory, never blocking: any failure
leaves the case without facets and the prose-fallback fingerprint applies.

Merge policy is fill-only: extraction never overwrites an existing value,
so a manual PUT /cases/{id}/facets override always wins.
"""

import asyncio
import os
from typing import Any, Dict, Optional

from facets import facets_complete, normalize_facets

FACET_EXTRACTION_MODEL = (
    os.getenv(
        "FACET_EXTRACTION_MODEL",
        os.getenv(
            "MEMORY_EXTRACTION_MODEL",
            os.getenv("LLAMA_SERVER_MODEL", "qwen3.6-27b-udq5xl-vision"),
        ),
    ).strip()
    or "qwen3.6-27b-udq5xl-vision"
)
FACET_EXTRACTION_NUM_CTX = int(
    (os.getenv("FACET_EXTRACTION_NUM_CTX", "32768") or "32768").strip()
)
FACET_EXTRACTION_RETRIES = int((os.getenv("FACET_EXTRACTION_RETRIES", "2") or "2").strip())
FACET_EXTRACTION_MAX_CHARS = int(
    (os.getenv("FACET_EXTRACTION_MAX_CHARS", "40000") or "40000").strip()
)
FACETS_ENABLED = (
    os.getenv("JURIS_FACETS_ENABLED", "true").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Flat on purpose; the profil axes are reshaped into the nested block in code.
_FACET_JSON_SPEC = """{
  "herkunftsland": "string|null",
  "staatsangehoerigkeit": "string|null",
  "verfahrensart": "asyl_klage|asyl_eilverfahren|asyl_folgeantrag|dublin|widerruf|aufenthaltsrecht|einbuergerung|abschiebungshaft|sonstiges|null",
  "schutzgruende": ["string"],
  "themen": ["string"],
  "region": "string|null",
  "alter": "number|null",
  "geschlecht": "m|w|d|null",
  "gesundheit": "string|null",
  "familienstand": "string|null",
  "netzwerk_im_herkunftsland": true,
  "besonderheiten": ["string"]
}"""

_FACET_EXTRACTION_RULES = """Du extrahierst die Kernmerkmale (Facetten) eines deutschen Asyl-/Migrationsrechtsfalls aus Aktendokumenten (v.a. dem Bescheid).

Regeln:
- Nur belegte Angaben übernehmen; Unbekanntes als null bzw. leere Liste.
- herkunftsland: Staat als deutsches Substantiv ("Syrien", "Afghanistan").
- schutzgruende: einschlägige Normen mit Gesetz zuerst ("AsylG § 4", "AufenthG § 60 Abs. 5").
- themen: knappe deutsche Schlagwörter klein geschrieben ("existenzminimum", "abschiebungsverbot", "netzwerk", "rückkehr", "krankheit").
- alter: Alter der Klagepartei in Jahren (Zahl, unbekannt: null), geschlecht: m/w/d.
- gesundheit: knapp ("gesund" oder Diagnose), familienstand: knapp ("ledig", "verheiratet, 2 Kinder").
- netzwerk_im_herkunftsland: true/false — hat die Klagepartei dort noch Familie/tragfähige Kontakte?
- besonderheiten: kurze Freitext-Merkmale ("ausreise als kind", "11 jahre jordanien").

Antworte NUR mit diesem JSON-Objekt:
""" + _FACET_JSON_SPEC


_PROFIL_KEYS = (
    "alter", "geschlecht", "gesundheit", "familienstand",
    "netzwerk_im_herkunftsland", "besonderheiten",
)


def facets_from_flat(parsed: Any) -> Dict[str, Any]:
    """Reshape the flat extraction JSON into the nested raw facet block
    (profil axes fold under "profil"); drops nulls. Not yet normalized."""
    if not isinstance(parsed, dict):
        return {}
    raw = {
        k: v for k, v in parsed.items()
        if k not in _PROFIL_KEYS and v not in (None, "", [], {})
    }
    profil = {
        k: parsed.get(k) for k in _PROFIL_KEYS
        if parsed.get(k) not in (None, "", [], {})
    }
    if profil:
        raw["profil"] = profil
    return raw


def merge_facets_fill_only(existing: Optional[Dict[str, Any]], extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Fill gaps only: an existing value (incl. manual overrides) always wins.
    The profil block merges per axis."""
    merged = dict(existing or {})
    for key, value in (extracted or {}).items():
        if key == "profil":
            profil = dict(merged.get("profil") or {})
            for axis, axis_value in (value or {}).items():
                profil.setdefault(axis, axis_value)
            if profil:
                merged["profil"] = profil
        elif key not in merged:
            merged[key] = value
    return merged


async def _qwen_json(prompt: str) -> Dict[str, Any]:
    """The one impure seam (stubbed in tests): service readiness + Qwen call."""
    from citation_qwen import call_qwen_json
    from shared import ensure_anonymization_service_ready

    await ensure_anonymization_service_ready()
    return await call_qwen_json(
        os.environ["ANONYMIZATION_SERVICE_URL"],
        prompt,
        model=FACET_EXTRACTION_MODEL,
        num_predict=1200,
        temperature=0.0,
        num_ctx=FACET_EXTRACTION_NUM_CTX,
    )


async def extract_facets_from_text(text: str) -> Dict[str, Any]:
    """One small Qwen call → normalized canonical facet block ({} on any
    failure — advisory, never raises). An empty answer is final: at
    temperature 0 with a warm prompt cache a retry returns the identical
    emptiness (e.g. a Jobcenter Akte with no asylum facets); only
    transport/service errors are retried."""
    if not os.environ.get("ANONYMIZATION_SERVICE_URL") or not (text or "").strip():
        return {}

    prompt = f"{_FACET_EXTRACTION_RULES}\n\nQUELLEN:\n{text[:FACET_EXTRACTION_MAX_CHARS]}"
    for attempt in range(1, FACET_EXTRACTION_RETRIES + 1):
        try:
            parsed = await _qwen_json(prompt)
            if parsed:
                return normalize_facets(facets_from_flat(parsed))
            print(f"[INFO] Facet extraction: nichts extrahiert (leere Antwort — kein Asyl-Material?)")
            return {}
        except Exception as exc:
            print(f"[WARN] Facet extraction failed (attempt {attempt}): {exc}")
        if attempt < FACET_EXTRACTION_RETRIES:
            await asyncio.sleep(2 * attempt)
    return {}


async def maybe_update_case_facets(db: Any, case: Any, material: str) -> Optional[Dict[str, Any]]:
    """Intake hook: extract + fill-only merge facets onto the case. Keeps
    running on later documents until the block is COMPLETE (a sparse first
    hit from a cover letter must not freeze it before the Bescheid arrives);
    existing values are never overwritten. Skips silently when disabled,
    complete, or extraction yields nothing new. Returns the stored block
    when it changed, else None."""
    if not FACETS_ENABLED or case is None:
        return None
    from rechtsgebiete import uses_asyl_layers

    gebiete = getattr(case, "rechtsgebiete", None) or getattr(case, "rechtsgebiet", None)
    if not uses_asyl_layers(gebiete):
        return None
    existing = case.facets_json or {}
    if facets_complete(existing):
        return None
    extracted = await extract_facets_from_text(material)
    if not extracted:
        return None
    merged = merge_facets_fill_only(existing, extracted)
    if merged == existing:
        return None
    try:
        case.facets_json = merged
        db.add(case)
        db.commit()
        print(f"[FACETS] Case {case.id}: facets extracted at intake: {sorted(merged.keys())}")
        return merged
    except Exception as exc:
        db.rollback()
        print(f"[WARN] Facet store failed for case {getattr(case, 'id', '?')}: {exc}")
        return None
