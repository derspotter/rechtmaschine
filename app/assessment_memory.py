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
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

ASSESSMENT_TARGET = "case_assessment"
ASSESSMENT_LIST_FIELDS = {"gutachten"}
ASSESSMENT_SCALAR_FIELDS = {"notizen"}
SERVER_FIELDS = ("store", "store_entry_id", "store_checked_at")

MAX_GUTACHTEN_BYTES = 24_000
MAX_CONTENT_BYTES = 200_000

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

StoreState = Literal["verified", "date_mismatch", "not_in_store", "unchecked"]


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
        if not _ISO_DATE_RE.match(value or ""):
            raise ValueError("datum muss ISO-Format JJJJ-MM-TT haben")
        return value


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
        if not _ISO_DATE_RE.match(value or ""):
            raise ValueError("stand muss ISO-Format JJJJ-MM-TT haben")
        return value

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


def validate_assessment_content(content: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize assessment content. Raises ValueError on any
    schema, uniqueness, cross-reference or size violation."""
    try:
        model = CaseAssessmentContent(**(content or {}))
    except Exception as exc:  # pydantic ValidationError
        raise ValueError(f"Ungueltiger Gutachten-Inhalt: {exc}") from exc

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
                        f"Pruefungspunkt verweist auf unbekanntes Az: {az}"
                    )

        if _json_size(entry) > MAX_GUTACHTEN_BYTES:
            raise ValueError(
                f"Gutachten {entry['id']} ueberschreitet {MAX_GUTACHTEN_BYTES} Bytes"
            )

    if _json_size(dumped) > MAX_CONTENT_BYTES:
        raise ValueError(f"Gutachten-Inhalt ueberschreitet {MAX_CONTENT_BYTES} Bytes")

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
        "Gutachten werden ueber /gutachten/by-id/<id> adressiert, nicht ueber den Index"
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
                    f"id im Pfad ({selector}) und im Wert ({value.get('id')}) stimmen nicht ueberein"
                )
            patched["gutachten"][_index_of(patched, selector)] = value
            continue

        if operation == "remove":
            if not selector or selector == "-":
                raise ValueError("remove verlangt den Pfad /gutachten/by-id/<id>")
            del patched["gutachten"][_index_of(patched, selector)]
            continue

        raise ValueError(f"Nicht unterstuetzte Operation: {operation}")

    return validate_assessment_content(patched)
