"""Domain logic for the case_assessment memory target (Gutachten).

Everything specific to legal assessments lives here: content models,
id-addressed patch ops, store reconciliation, prompt rendering and the
rebase strategy. The generic memory layer (agent_memory_service) keeps
persistence, revisions, versions and the proposal lifecycle.
"""
from __future__ import annotations

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
