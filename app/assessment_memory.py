"""Domain logic for the case_assessment memory target (Gutachten).

Everything specific to legal assessments lives here: content models,
id-addressed patch ops, store reconciliation, prompt rendering and the
rebase strategy. The generic memory layer (agent_memory_service) keeps
persistence, revisions, versions and the proposal lifecycle.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from citation_identity import canonical_az

ASSESSMENT_TARGET = "case_assessment"
ASSESSMENT_LIST_FIELDS = {"gutachten"}
ASSESSMENT_SCALAR_FIELDS = {"notizen"}
SERVER_FIELDS = ("store", "store_entry_id", "store_checked_at")

MAX_GUTACHTEN_BYTES = 24_000
MAX_CONTENT_BYTES = 200_000

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

StoreState = Literal["verified", "date_mismatch", "not_in_store", "unchecked"]


def _require_iso_date(value: str, field: str) -> str:
    """ISO-Datum JJJJ-MM-TT, das es im Kalender wirklich gibt.

    Die Regex allein lässt 2026-02-31 durch -- ein solches Datum kann der
    Store-Abgleich nie treffen und der Prompt zeigte es als echtes
    Entscheidungsdatum an.
    """
    if not _ISO_DATE_RE.match(value or ""):
        raise ValueError(f"{field} muss ISO-Format JJJJ-MM-TT haben")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{field} ist kein gültiges Kalenderdatum: {value}") from None
    return value


class Fundstelle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gericht: str = Field(min_length=1, max_length=120)
    datum: str
    az: str = Field(min_length=1, max_length=80)
    art: Optional[Literal["Urteil", "Beschluss"]] = None
    aussage: str = Field(default="", max_length=400)
    richtung: Literal["pro", "contra", "neutral"] = "neutral"
    store: StoreState = "unchecked"
    store_entry_id: Optional[str] = None
    store_checked_at: Optional[str] = None

    @field_validator("datum")
    @classmethod
    def _iso(cls, value: str) -> str:
        return _require_iso_date(value, "datum")


class PruefungsPunkt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    these: str = Field(min_length=1, max_length=300)
    bewertung: str = Field(default="", max_length=800)
    fundstellen: List[str] = Field(default_factory=list)


class GutachtenEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    rechtsfrage: str = Field(min_length=1, max_length=300)
    ergebnis: str = Field(min_length=1, max_length=800)
    stand: str
    status: Literal["aktiv", "ueberholt"] = "aktiv"
    pruefung: List[PruefungsPunkt] = Field(default_factory=list, max_length=12)
    fundstellen: List[Fundstelle] = Field(default_factory=list, max_length=25)
    risiken: List[str] = Field(default_factory=list, max_length=10)
    quelle: str = Field(default="", max_length=200)

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not _SLUG_RE.match(value or ""):
            raise ValueError(
                "id muss ein Slug aus Kleinbuchstaben, Ziffern und Bindestrichen sein"
            )
        return value

    @field_validator("stand")
    @classmethod
    def _iso(cls, value: str) -> str:
        return _require_iso_date(value, "stand")

    @field_validator("risiken")
    @classmethod
    def _risk_len(cls, value: List[str]) -> List[str]:
        for item in value:
            if len(item) > 300:
                raise ValueError("Risiko-Eintrag ist zu lang (max 300 Zeichen)")
        return value


class CaseAssessmentContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gutachten: List[GutachtenEntry] = Field(default_factory=list)
    notizen: str = ""


def default_case_assessment_json() -> Dict[str, Any]:
    return CaseAssessmentContent().model_dump(mode="json")


def strip_server_fields(value: Any) -> Any:
    """Recursively drop server-owned Fundstelle fields from a client value."""
    if isinstance(value, dict):
        return {
            key: strip_server_fields(item)
            for key, item in value.items()
            if key not in SERVER_FIELDS
        }
    if isinstance(value, list):
        return [strip_server_fields(item) for item in value]
    return value


def _json_size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _without_server_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy of a dumped Gutachten entry with the server-owned Fundstelle
    fields removed -- used to measure size without penalizing a Gutachten
    for the store-reconciliation state the server itself writes onto it."""
    e = copy.deepcopy(entry)
    for f in e.get("fundstellen") or []:
        for k in SERVER_FIELDS:
            f.pop(k, None)
    return e


def validate_assessment_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize assessment content. Raises ValueError on any
    schema, uniqueness, cross-reference or size violation.

    Size is measured on a copy with the three server-owned Fundstelle fields
    (store, store_entry_id, store_checked_at) stripped -- a Gutachten right
    at the client-authored budget must not start failing validation the
    moment the server stamps its store state onto it."""
    try:
        model = CaseAssessmentContent(**(content or {}))
    except Exception as exc:  # pydantic ValidationError
        raise ValueError(f"Ungültiger Gutachten-Inhalt: {exc}") from exc

    dumped = model.model_dump(mode="json")

    seen: set = set()
    for entry in dumped["gutachten"]:
        if entry["id"] in seen:
            raise ValueError(f"Doppelte Gutachten-id: {entry['id']}")
        seen.add(entry["id"])

        known_az = {f["az"] for f in entry["fundstellen"]}
        for punkt in entry["pruefung"]:
            for az in punkt["fundstellen"]:
                if az not in known_az:
                    raise ValueError(
                        f"Prüfungspunkt verweist auf unbekanntes Az: {az}"
                    )

        seen_az: set = set()
        for f in entry["fundstellen"]:
            key = canonical_az(f["az"])
            if key in seen_az:
                raise ValueError(
                    f"Gutachten {entry['id']}: Fundstelle {f['az']} ist doppelt"
                )
            seen_az.add(key)

        if _json_size(_without_server_fields(entry)) > MAX_GUTACHTEN_BYTES:
            raise ValueError(
                f"Gutachten {entry['id']} überschreitet {MAX_GUTACHTEN_BYTES} Bytes"
            )

    stripped_total = {
        "gutachten": [_without_server_fields(e) for e in dumped["gutachten"]],
        "notizen": dumped.get("notizen", ""),
    }
    if _json_size(stripped_total) > MAX_CONTENT_BYTES:
        raise ValueError(f"Gutachten-Inhalt überschreitet {MAX_CONTENT_BYTES} Bytes")

    return dumped


_BY_ID_PREFIX = "by-id"


def sanitize_assessment_ops(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a copy of ops with server-owned Fundstelle fields removed.

    Proposals are stored verbatim, so the stripping has to happen before the
    ops are persisted -- otherwise a client-supplied store="verified" would
    survive into the accepted content."""
    cleaned: List[Dict[str, Any]] = []
    for op in ops:
        item = dict(op)
        if "value" in item:
            item["value"] = strip_server_fields(item["value"])
        cleaned.append(item)
    return cleaned


def _parse_assessment_path(path: str) -> tuple:
    parts = [p for p in str(path or "").split("/") if p != ""]
    if not parts:
        raise ValueError("Patch-Pfad fehlt")
    field = parts[0]
    if field == "notizen":
        if len(parts) != 1:
            raise ValueError("Pfad /notizen erlaubt keine Unterpfade")
        return ("notizen", None)
    if field != "gutachten":
        raise ValueError(f"Patch-Pfad ist nicht erlaubt: /{field}")
    if len(parts) == 1:
        return ("gutachten", None)
    if parts[1] == "-":
        return ("gutachten", "-")
    if parts[1] == _BY_ID_PREFIX and len(parts) == 3:
        return ("gutachten", parts[2])
    raise ValueError(
        "Gutachten werden über /gutachten/by-id/<id> adressiert, nicht über den Index"
    )


def _index_of(content: Dict[str, Any], gutachten_id: str) -> int:
    for index, entry in enumerate(content.get("gutachten") or []):
        if entry.get("id") == gutachten_id:
            return index
    raise ValueError(f"Unbekannte Gutachten-id: {gutachten_id}")


def apply_assessment_ops(
    content: Dict[str, Any], ops: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply id-addressed ops and return validated content."""
    patched = copy.deepcopy(content or {})
    patched.setdefault("gutachten", [])
    patched.setdefault("notizen", "")

    if not ops:
        raise ValueError("Patch braucht mindestens eine Operation")

    for op in ops:
        operation = op.get("op")
        field, selector = _parse_assessment_path(op.get("path"))
        value = strip_server_fields(op.get("value"))

        if field == "notizen":
            if operation != "set":
                raise ValueError("Auf /notizen ist nur set erlaubt")
            patched["notizen"] = value
            continue

        if operation == "append":
            if selector != "-":
                raise ValueError("append verlangt den Pfad /gutachten/-")
            if not isinstance(value, dict):
                raise ValueError("append verlangt ein Gutachten-Objekt")
            new_id = value.get("id")
            if any(e.get("id") == new_id for e in patched["gutachten"]):
                raise ValueError(
                    f"Gutachten {new_id} existiert bereits, bitte set /gutachten/by-id/{new_id}"
                )
            patched["gutachten"].append(value)
            continue

        if operation == "set":
            if not selector or selector == "-":
                raise ValueError("set verlangt den Pfad /gutachten/by-id/<id>")
            if not isinstance(value, dict):
                raise ValueError("set verlangt ein Gutachten-Objekt")
            if value.get("id") != selector:
                raise ValueError(
                    f"id im Pfad ({selector}) und im Wert ({value.get('id')}) stimmen nicht überein"
                )
            patched["gutachten"][_index_of(patched, selector)] = value
            continue

        if operation == "remove":
            if not selector or selector == "-":
                raise ValueError("remove verlangt den Pfad /gutachten/by-id/<id>")
            del patched["gutachten"][_index_of(patched, selector)]
            continue

        raise ValueError(f"Nicht unterstützte Operation: {operation}")

    return validate_assessment_content(patched)


ASSESSMENT_BLOCK_HEADER = (
    "RECHTLICHE WÜRDIGUNG DER KANZLEI "
    "(Fundstellen mit Store-Abgleich, Stand je Gutachten):"
)


def _de_date(iso: str) -> str:
    year, month, day = iso.split("-")
    return f"{day}.{month}.{year}"


def _render_entry(entry: Dict[str, Any], with_pruefung: bool, with_risiken: bool) -> str:
    """Render one Gutachten. Rechtsfrage, Ergebnis and the blocklist line are
    unconditional -- only pruefung and risiken are subject to the budget
    stages in render_assessment_block."""
    verified = {
        f["az"]: f for f in entry.get("fundstellen") or [] if f.get("store") == "verified"
    }
    blocked = [
        f["az"] for f in entry.get("fundstellen") or [] if f.get("store") != "verified"
    ]
    lines = [
        f"[{entry['id']}, Stand {_de_date(entry['stand'])}] "
        f"Rechtsfrage: {entry['rechtsfrage']} Ergebnis: {entry['ergebnis']}"
    ]
    if with_pruefung:
        for punkt in entry.get("pruefung") or []:
            cites = [
                f"{verified[az]['gericht']} {az} "
                f"({_de_date(verified[az]['datum'])}, {verified[az]['richtung']})"
                for az in punkt.get("fundstellen") or []
                if az in verified
            ]
            suffix = f" – Fundstellen: {', '.join(cites)}" if cites else ""
            lines.append(f"  Prüfung: {punkt['these']} – {punkt['bewertung']}{suffix}")
    if with_risiken and entry.get("risiken"):
        lines.append("  Risiken: " + ", ".join(entry["risiken"]))
    if blocked:
        lines.append("  Nicht zitierfähig (nicht im Bestand): " + ", ".join(blocked))
    return "\n".join(lines)


def _assessment_used_and_blocked(entries: List[Dict[str, Any]]) -> tuple:
    used = [e["id"] for e in entries]
    blocked = [
        f["az"]
        for e in entries
        for f in e.get("fundstellen") or []
        if f.get("store") != "verified"
    ]
    return used, blocked


# 4000 reichte auf der ersten echten Akte nicht: zwei Gutachten (5.6k Zeichen)
# fielen auf Stufe 3 zurück und der Cloud-Prompt trug keine einzige Fundstelle
# (Live-Abnahme 157/26, 07.09.2026). 12000 Zeichen sind ~3.5k Tokens.
ASSESSMENT_BLOCK_MAX_CHARS = 12_000


def render_assessment_block(content: Dict[str, Any], max_chars: int = ASSESSMENT_BLOCK_MAX_CHARS) -> tuple:
    """Render the prompt block. Returns (text, used_ids, blocked_az).

    Only `aktiv` Gutachten are rendered, newest `stand` first. Truncation
    proceeds in stages against the whole Gutachten list, oldest last:
    1) everything (pruefung + risiken), 2) drop risiken, 3) drop pruefung too
    -- rechtsfrage, ergebnis and the blocklist line always survive. Only once
    stage 3 still does not fit are whole Gutachten dropped from the end
    (never a half Gutachten -- the last one standing is kept even if it alone
    still exceeds the budget)."""
    assessment = validate_assessment_content(content)
    active = [e for e in assessment["gutachten"] if e.get("status") == "aktiv"]
    if not active:
        return "", [], []
    active.sort(key=lambda e: e["stand"], reverse=True)

    def build(entries: List[Dict[str, Any]], with_pruefung: bool, with_risiken: bool) -> str:
        body = "\n".join(_render_entry(e, with_pruefung, with_risiken) for e in entries)
        dropped = len(active) - len(entries)
        if dropped:
            body += f"\n[weitere Gutachten gekürzt: {dropped}]"
        return f"{ASSESSMENT_BLOCK_HEADER}\n{body}"

    # Stages 1-2: the full Gutachten list, only dropping fields.
    for with_pruefung, with_risiken in ((True, True), (True, False)):
        text = build(active, with_pruefung, with_risiken)
        if len(text) <= max_chars:
            used, blocked = _assessment_used_and_blocked(active)
            return text, used, blocked

    # Stage 3: rechtsfrage/ergebnis/blocklist only, dropping whole Gutachten
    # from the end until it fits.
    entries = list(active)
    while True:
        text = build(entries, False, False)
        if len(text) <= max_chars or len(entries) == 1:
            used, blocked = _assessment_used_and_blocked(entries)
            return text, used, blocked
        entries = entries[:-1]


def render_case_assessment_compact(content: Dict[str, Any]) -> str:
    """search_text renderer for the ORM row (no budget)."""
    text, _, _ = render_assessment_block(content, max_chars=MAX_CONTENT_BYTES)
    return text or "Rechtliche Würdigung: Keine gepflegten Inhalte."


def load_store_map(db: Any) -> Dict[str, List[Dict[str, Any]]]:
    """One pass over the active Rechtsprechung entries, keyed by normalized Az.

    store_lookup() rescans the whole table per call -- for a Gutachten with 25
    Fundstellen that would be 25 full scans, and the nightly recheck multiplies
    that by every case. Load once, compare in memory."""
    from models import RechtsprechungEntry
    from verify_source import az_for_compare

    store_map: Dict[str, List[Dict[str, Any]]] = {}
    rows = (
        db.query(RechtsprechungEntry)
        .filter(
            RechtsprechungEntry.is_active.is_(True),
            RechtsprechungEntry.aktenzeichen.isnot(None),
        )
        .all()
    )
    for row in rows:
        key = az_for_compare(row.aktenzeichen)
        if not key:
            continue
        store_map.setdefault(key, []).append(
            {
                "id": str(row.id),
                "decision_date": row.decision_date.isoformat() if row.decision_date else None,
            }
        )
    return store_map


def _content_without_server_fields(entry: Dict[str, Any]) -> str:
    return json.dumps(strip_server_fields(entry), ensure_ascii=False, sort_keys=True)


def changed_gutachten_ids(previous: Dict[str, Any], new: Dict[str, Any]) -> set:
    """Ids whose substance changed, ignoring server-owned fields."""
    before = {
        e["id"]: _content_without_server_fields(e)
        for e in (previous or {}).get("gutachten") or []
    }
    after = {
        e["id"]: _content_without_server_fields(e)
        for e in (new or {}).get("gutachten") or []
    }
    changed = {gid for gid, body in after.items() if before.get(gid) != body}
    changed |= set(before) - set(after)
    return changed


def reconcile_store(
    content: Dict[str, Any],
    store_map: Dict[str, List[Dict[str, Any]]],
    changed_ids: Optional[set],
    now_iso: Optional[str] = None,
) -> tuple:
    """Set the server-owned store fields. Returns (content, warnings).

    changed_ids=None means every Gutachten is reconciled (recheck). A set
    limits the work to the Gutachten an accept actually touched."""
    from verify_source import az_for_compare

    stamp = now_iso or datetime.utcnow().isoformat()
    patched = copy.deepcopy(content or {})
    warnings: List[Dict[str, str]] = []

    for entry in patched.get("gutachten") or []:
        if changed_ids is not None and entry.get("id") not in changed_ids:
            continue
        for fundstelle in entry.get("fundstellen") or []:
            key = az_for_compare(fundstelle.get("az"))
            hits = store_map.get(key) or []
            match = next(
                (h for h in hits if h.get("decision_date") == fundstelle.get("datum")),
                None,
            )
            if match:
                state, entry_id = "verified", match["id"]
            elif hits:
                state, entry_id = "date_mismatch", None
            else:
                state, entry_id = "not_in_store", None
            fundstelle["store"] = state
            fundstelle["store_entry_id"] = entry_id
            fundstelle["store_checked_at"] = stamp
            if state != "verified":
                warnings.append(
                    {
                        "gutachten_id": entry.get("id"),
                        "az": fundstelle.get("az"),
                        "store": state,
                    }
                )

    return validate_assessment_content(patched), warnings


def citation_lines(warnings: List[Dict[str, str]], content: Dict[str, Any]) -> List[str]:
    """Parser-compatible citation lines for every not_in_store Fundstelle."""
    wanted = {
        (w["gutachten_id"], w["az"]) for w in warnings if w["store"] == "not_in_store"
    }
    lines: List[str] = []
    for entry in content.get("gutachten") or []:
        for fundstelle in entry.get("fundstellen") or []:
            if (entry.get("id"), fundstelle.get("az")) not in wanted:
                continue
            year, month, day = fundstelle["datum"].split("-")
            art = fundstelle.get("art") or "Beschluss"
            lines.append(
                f"{fundstelle['gericht']}, {art} vom {day}.{month}.{year} – {fundstelle['az']}"
            )
    return lines


def _op_target_id(op: Dict[str, Any]) -> Optional[str]:
    """Extract the Gutachten id from an operation, or None if not a gutachten op."""
    try:
        field, selector = _parse_assessment_path(op.get("path"))
    except ValueError:
        return None
    if field != "gutachten":
        return None
    if selector and selector != "-":
        return selector
    value = op.get("value")
    return value.get("id") if isinstance(value, dict) else None


def touched_gutachten_ids(ops: List[Dict[str, Any]]) -> set:
    """Ids that an op set addresses (set/remove by-id) or creates (append)."""
    ids: set = set()
    for op in ops or []:
        if isinstance(op, dict):
            gid = _op_target_id(op)
            if gid:
                ids.add(gid)
    return ids


def rebase_assessment_ops(
    ops: List[Dict[str, Any]],
    new_content: Dict[str, Any],
    conflict_ids: set,
) -> List[Dict[str, Any]]:
    """Alles oder nichts: entweder alle Ops überleben oder das Proposal wird
    superseded.

    Identity is the Gutachten id. The whole proposal is dropped (returns [])
    when any op targets an id in the accept's conflict set, when the
    /notizen op conflicts, or when the ops -- applied together, in order --
    no longer apply cleanly to the new content."""
    ops = [op for op in ops if isinstance(op, dict)]
    for op in ops:
        gid = _op_target_id(op)
        if gid is None:
            if str(op.get("path") or "").strip("/") == "notizen" and "notizen" in conflict_ids:
                return []
            continue
        if gid in conflict_ids:
            return []
    try:
        apply_assessment_ops(new_content, ops)
    except ValueError:
        return []
    return ops


def count_store_changes(previous: Dict[str, Any], new: Dict[str, Any]) -> tuple:
    """Count how many Fundstellen (and their parent Gutachten) changed store state.

    Compares, per Gutachten id and per Az, the pair `(store, store_entry_id)`
    between `previous` and `new`. A change in either field counts. Pure
    function, no DB access -- used by `recheck_assessment` and independently
    testable."""
    changed_fundstellen = 0
    changed_gutachten: set = set()
    old_by_id = {e["id"]: e for e in previous.get("gutachten") or []}
    for entry in new.get("gutachten") or []:
        old_entry = old_by_id.get(entry["id"], {})
        old_states = {
            f["az"]: (f.get("store"), f.get("store_entry_id"))
            for f in old_entry.get("fundstellen") or []
        }
        for fundstelle in entry.get("fundstellen") or []:
            now_state = (fundstelle.get("store"), fundstelle.get("store_entry_id"))
            if old_states.get(fundstelle["az"]) != now_state:
                changed_fundstellen += 1
                changed_gutachten.add(entry["id"])
    return changed_fundstellen, len(changed_gutachten)


def recheck_assessment(db: Any, owner_id: Any, case_id: Any) -> Dict[str, Any]:
    """Re-run the store reconciliation for every Fundstelle of a case.

    Also re-checks entries that are already `verified`: a deactivated or
    corrected store entry must not stay quotable forever. The re-stamped
    content is always persisted -- store_checked_at must be current even on
    a Fundstelle whose state did not change, otherwise a recheck that
    confirms "still verified" leaves a stale timestamp. A revision is only
    written, and does NOT bump the version, when a store STATE actually
    changed; pending proposals are never rebased here -- only server-owned
    fields change, and proposal ops never carry those. A case without an
    active Gutachten returns early with `skipped`, before the store is read."""
    from agent_memory_service import (
        ASSESSMENT_TARGET,
        _create_revision,
        _target_content,
        _write_target_content,
        get_or_create_case_assessment,
        render_case_assessment_compact,
    )

    target = get_or_create_case_assessment(db, owner_id, case_id, for_update=True)
    previous = _target_content(ASSESSMENT_TARGET, target)
    # Ohne aktives Gutachten gibt es nichts abzugleichen: kein Store-Scan,
    # keine Revision -- der Lauf über alle Akten spart damit die Mehrzahl der
    # Fälle ein.
    if not [e for e in (previous.get("gutachten") or []) if e.get("status") == "aktiv"]:
        return {
            "skipped": "keine Gutachten",
            "changed_fundstellen": 0,
            "changed_gutachten": 0,
            "warnings": [],
        }

    store_map = load_store_map(db)
    new_content, warnings = reconcile_store(previous, store_map, None)

    changed_fundstellen, changed_gutachten_count = count_store_changes(previous, new_content)

    if changed_fundstellen:
        _create_revision(db, ASSESSMENT_TARGET, target, previous, new_content, [], "recheck")
    _write_target_content(ASSESSMENT_TARGET, target, new_content)  # store_checked_at immer aktuell
    target.search_text = render_case_assessment_compact(new_content)
    target.updated_at = datetime.utcnow()
    db.add(target)
    db.commit()

    return {
        "changed_fundstellen": changed_fundstellen,
        "changed_gutachten": changed_gutachten_count,
        "warnings": warnings,
    }


WIKI_MAX_CHARS = 24_000


def verified_az_whitelist(
    content: Dict[str, Any], only_ids: Optional[set] = None
) -> set:
    """Normalized Az of every verified Fundstelle in active Gutachten.

    `only_ids`, when given, restricts the whitelist to Gutachten whose id is
    in the set -- pair with render_assessment_for_wiki's rendered_ids so a
    Gutachten cut from the wiki text for size no longer licenses its own
    citations there."""
    assessment = validate_assessment_content(content)
    return {
        canonical_az(f["az"])
        for entry in assessment["gutachten"]
        if entry.get("status") == "aktiv"
        and (only_ids is None or entry.get("id") in only_ids)
        for f in entry.get("fundstellen") or []
        if f.get("store") == "verified"
    }


def render_assessment_for_wiki(
    content: Dict[str, Any], max_chars: int = WIKI_MAX_CHARS
) -> tuple:
    """Full-detail rendering for the wiki distillation, verified citations only,
    in the format the citation parser understands.

    Returns (text, rendered_ids): rendered_ids are the ids of the Gutachten
    whose block actually made it into `text`. A Gutachten dropped for
    budget must not still license its Fundstellen via verified_az_whitelist,
    so callers pass rendered_ids on as that function's only_ids."""
    assessment = validate_assessment_content(content)
    active = [e for e in assessment["gutachten"] if e.get("status") == "aktiv"]
    if not active:
        return "", set()
    active.sort(key=lambda e: e["stand"], reverse=True)

    blocks: List[tuple] = []
    for entry in active:
        verified = {
            f["az"]: f
            for f in entry.get("fundstellen") or []
            if f.get("store") == "verified"
        }
        lines = [
            f"GUTACHTEN {entry['id']} (Stand {_de_date(entry['stand'])})",
            f"Rechtsfrage: {entry['rechtsfrage']}",
            f"Ergebnis: {entry['ergebnis']}",
        ]
        for punkt in entry.get("pruefung") or []:
            lines.append(f"- These: {punkt['these']}")
            lines.append(f"  Bewertung: {punkt['bewertung']}")
            for az in punkt.get("fundstellen") or []:
                f = verified.get(az)
                if not f:
                    continue
                art = f.get("art") or "Beschluss"
                lines.append(
                    f"  Fundstelle ({f['richtung']}): {f['gericht']}, {art} vom "
                    f"{_de_date(f['datum'])} – {f['az']} — {f['aussage']}"
                )
        if entry.get("risiken"):
            lines.append("Risiken: " + ", ".join(entry["risiken"]))
        blocks.append((entry["id"], "\n".join(lines)))

    text = "\n\n".join(block for _, block in blocks)
    if len(text) <= max_chars:
        return text, {gid for gid, _ in blocks}

    kept: List[tuple] = []
    size = 0
    for gid, block in blocks:
        if size + len(block) > max_chars:
            continue  # skip this oversized Gutachten, smaller ones may still fit
        kept.append((gid, block))
        size += len(block) + 2
    text = "\n\n".join(block for _, block in kept)
    dropped = len(blocks) - len(kept)
    if dropped:
        text += f"\n\n[weitere Gutachten gekürzt: {dropped}]"
    return text, {gid for gid, _ in kept}
