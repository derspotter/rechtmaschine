from __future__ import annotations

import copy
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

import models as orm_models
from shared import (
    CaseBriefContent,
    CaseStrategyContent,
    MemoryPatchOperation,
    MemorySourceRef,
    MemoryTargetType,
)


BRIEF_TARGET = "case_brief"
STRATEGY_TARGET = "case_strategy"

BRIEF_LIST_FIELDS = {
    "beteiligte",
    "verfahrensstand",
    "sachverhalt",
    "antraege_ziele",
    "streitige_punkte",
    "beweismittel",
    "risiken",
    "offene_fragen",
}
BRIEF_SCALAR_FIELDS = {"notizen"}

STRATEGY_LIST_FIELDS = {
    "argumentationslinien",
    "rechtliche_ansatzpunkte",
    "beweisstrategie",
    "prozessuale_schritte",
    "vergleich_oder_taktik",
    "risiken_und_gegenargumente",
    "offene_fragen",
}
STRATEGY_SCALAR_FIELDS = {"kernstrategie", "notizen"}


def _model(name: str) -> Any:
    model = getattr(orm_models, name, None)
    if model is None:
        raise RuntimeError(f"Memory ORM model {name} is not available")
    return model


def _model_dump(value: Any) -> Dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value or {})


def _uuid(value: Any, field_name: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name}") from exc


def _column_names(model: Any) -> set[str]:
    table = getattr(model, "__table__", None)
    if table is None:
        return set()
    return {column.key for column in table.columns}


def _new_model(model: Any, values: Dict[str, Any]) -> Any:
    columns = _column_names(model)
    filtered = {key: value for key, value in values.items() if key in columns}
    return model(**filtered)


def _set_if_column(row: Any, key: str, value: Any) -> None:
    if key in _column_names(type(row)):
        setattr(row, key, value)


def _has_column(row_or_model: Any, key: str) -> bool:
    model = row_or_model if isinstance(row_or_model, type) else type(row_or_model)
    return key in _column_names(model)


def _metadata_value(row: Any) -> Dict[str, Any]:
    for attr in ("metadata_", "metadata_json", "extra_json"):
        value = getattr(row, attr, None)
        if isinstance(value, dict):
            return value
    return {}


def _source_refs_to_dicts(source_refs: Iterable[Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for source_ref in source_refs or []:
        ref = _model_dump(source_ref)
        if not ref.get("source_type"):
            raise ValueError("source_refs must include source_type")
        refs.append(ref)
    return refs


def _validate_brief_content(content: Dict[str, Any]) -> Dict[str, Any]:
    allowed = BRIEF_LIST_FIELDS | BRIEF_SCALAR_FIELDS
    extra = set(content or {}) - allowed
    if extra:
        raise ValueError(f"Invalid case brief fields: {', '.join(sorted(extra))}")
    model = CaseBriefContent(**(content or {}))
    return _model_dump(model)


def _validate_strategy_content(content: Dict[str, Any]) -> Dict[str, Any]:
    allowed = STRATEGY_LIST_FIELDS | STRATEGY_SCALAR_FIELDS
    extra = set(content or {}) - allowed
    if extra:
        raise ValueError(f"Invalid case strategy fields: {', '.join(sorted(extra))}")
    model = CaseStrategyContent(**(content or {}))
    return _model_dump(model)


def default_case_brief_json() -> Dict[str, Any]:
    return _model_dump(CaseBriefContent())


def default_case_strategy_json() -> Dict[str, Any]:
    return _model_dump(CaseStrategyContent())


def _append_lines(lines: List[str], title: str, values: List[Any]) -> None:
    if not values:
        return
    rendered_values: List[str] = []
    for value in values:
        if isinstance(value, dict):
            label = value.get("name") or value.get("label") or value.get("rolle") or ""
            role = value.get("rolle") or value.get("role") or ""
            detail = value.get("notiz") or value.get("detail") or value.get("beschreibung") or ""
            rendered = " - ".join(part for part in (label, role, detail) if part)
            rendered_values.append(rendered or str(value))
        else:
            rendered_values.append(str(value))
    lines.append(f"{title}: " + "; ".join(item for item in rendered_values if item))


def render_case_brief_compact(content: Dict[str, Any]) -> str:
    brief = _validate_brief_content(content)
    lines: List[str] = ["Fallbrief:"]
    _append_lines(lines, "Beteiligte", brief["beteiligte"])
    _append_lines(lines, "Verfahrensstand", brief["verfahrensstand"])
    _append_lines(lines, "Sachverhalt", brief["sachverhalt"])
    _append_lines(lines, "Anträge/Ziele", brief["antraege_ziele"])
    _append_lines(lines, "Streitige Punkte", brief["streitige_punkte"])
    _append_lines(lines, "Beweismittel", brief["beweismittel"])
    _append_lines(lines, "Risiken", brief["risiken"])
    _append_lines(lines, "Offene Fragen", brief["offene_fragen"])
    if brief["notizen"].strip():
        lines.append(f"Notizen: {brief['notizen'].strip()}")
    return "\n".join(lines if len(lines) > 1 else ["Fallbrief: Keine gepflegten Inhalte."])


def render_case_strategy_compact(content: Dict[str, Any]) -> str:
    strategy = _validate_strategy_content(content)
    lines: List[str] = ["Fallstrategie:"]
    if strategy["kernstrategie"].strip():
        lines.append(f"Kernstrategie: {strategy['kernstrategie'].strip()}")
    _append_lines(lines, "Argumentationslinien", strategy["argumentationslinien"])
    _append_lines(lines, "Rechtliche Ansatzpunkte", strategy["rechtliche_ansatzpunkte"])
    _append_lines(lines, "Beweisstrategie", strategy["beweisstrategie"])
    _append_lines(lines, "Prozessuale Schritte", strategy["prozessuale_schritte"])
    _append_lines(lines, "Vergleich/Taktik", strategy["vergleich_oder_taktik"])
    _append_lines(lines, "Risiken/Gegenargumente", strategy["risiken_und_gegenargumente"])
    _append_lines(lines, "Offene Fragen", strategy["offene_fragen"])
    if strategy["notizen"].strip():
        lines.append(f"Notizen: {strategy['notizen'].strip()}")
    return "\n".join(lines if len(lines) > 1 else ["Fallstrategie: Keine gepflegten Inhalte."])


def _target_spec(target_type: MemoryTargetType) -> Tuple[Any, Any, str, Any, Any, set[str], set[str]]:
    if target_type == BRIEF_TARGET:
        return (
            _model("CaseBrief"),
            _model("CaseBriefSource"),
            "case_brief_id",
            default_case_brief_json,
            render_case_brief_compact,
            BRIEF_LIST_FIELDS,
            BRIEF_SCALAR_FIELDS,
        )
    if target_type == STRATEGY_TARGET:
        return (
            _model("CaseStrategy"),
            _model("CaseStrategySource"),
            "case_strategy_id",
            default_case_strategy_json,
            render_case_strategy_compact,
            STRATEGY_LIST_FIELDS,
            STRATEGY_SCALAR_FIELDS,
        )
    raise ValueError(f"Unsupported memory target type: {target_type}")


def _target_content(target_type: MemoryTargetType, target: Any) -> Dict[str, Any]:
    if _has_column(target, "content_json"):
        content = getattr(target, "content_json", None) or {}
        if target_type == BRIEF_TARGET:
            return _validate_brief_content(content)
        return _validate_strategy_content(content)

    if target_type == BRIEF_TARGET:
        return _validate_brief_content(
            {
                "beteiligte": getattr(target, "parties", None) or [],
                "verfahrensstand": getattr(target, "procedural_history", None) or [],
                "sachverhalt": getattr(target, "facts", None) or [],
                "antraege_ziele": getattr(target, "claims", None) or [],
                "risiken": getattr(target, "risks", None) or [],
                "offene_fragen": getattr(target, "open_questions", None) or [],
                "notizen": getattr(target, "summary", None) or "",
            }
        )

    return _validate_strategy_content(
        {
            "kernstrategie": getattr(target, "theory", None) or getattr(target, "objective", None) or "",
            "argumentationslinien": getattr(target, "arguments", None) or [],
            "beweisstrategie": getattr(target, "evidence_plan", None) or [],
            "prozessuale_schritte": getattr(target, "next_steps", None) or [],
            "risiken_und_gegenargumente": getattr(target, "risks", None) or [],
            "offene_fragen": getattr(target, "open_questions", None) or [],
            "notizen": getattr(target, "objective", None) or "",
        }
    )


def _write_target_content(target_type: MemoryTargetType, target: Any, content: Dict[str, Any]) -> None:
    if _has_column(target, "content_json"):
        target.content_json = content
        return

    if target_type == BRIEF_TARGET:
        _set_if_column(target, "parties", content.get("beteiligte", []))
        _set_if_column(target, "procedural_history", content.get("verfahrensstand", []))
        facts = list(content.get("sachverhalt", []))
        facts.extend(f"Streitiger Punkt: {item}" for item in content.get("streitige_punkte", []))
        facts.extend(f"Beweismittel: {item}" for item in content.get("beweismittel", []))
        _set_if_column(target, "facts", facts)
        _set_if_column(target, "claims", content.get("antraege_ziele", []))
        _set_if_column(target, "risks", content.get("risiken", []))
        _set_if_column(target, "open_questions", content.get("offene_fragen", []))
        _set_if_column(target, "summary", content.get("notizen", ""))
        return

    _set_if_column(target, "theory", content.get("kernstrategie", ""))
    _set_if_column(target, "objective", content.get("notizen", "") or content.get("kernstrategie", ""))
    _set_if_column(target, "arguments", content.get("argumentationslinien", []))
    _set_if_column(target, "evidence_plan", content.get("beweisstrategie", []))
    _set_if_column(target, "next_steps", content.get("prozessuale_schritte", []))
    _set_if_column(target, "risks", content.get("risiken_und_gegenargumente", []))
    _set_if_column(target, "open_questions", content.get("offene_fragen", []))


def _target_version(db: Session, target_type: MemoryTargetType, target: Any) -> int:
    if _has_column(target, "version"):
        return int(getattr(target, "version", 0) or 0)
    revision_model = _model("CaseMemoryRevision")
    count = (
        db.query(revision_model)
        .filter(
            revision_model.target_type == target_type,
            revision_model.target_id == getattr(target, "id", None),
        )
        .count()
    )
    return count + 1


def _get_target(db: Session, target_type: MemoryTargetType, owner_id: Any, target_id: Any) -> Any:
    model, _, _, _, _, _, _ = _target_spec(target_type)
    row = (
        db.query(model)
        .filter(model.id == _uuid(target_id, "target_id"), model.owner_id == _uuid(owner_id, "owner_id"))
        .first()
    )
    if not row:
        raise ValueError("Memory target not found")
    return row


def _get_or_create_target(
    db: Session,
    target_type: MemoryTargetType,
    owner_id: Any,
    case_id: Any,
) -> Any:
    model, _, _, default_factory, renderer, _, _ = _target_spec(target_type)
    owner_uuid = _uuid(owner_id, "owner_id")
    case_uuid = _uuid(case_id, "case_id")
    row = (
        db.query(model)
        .filter(model.owner_id == owner_uuid, model.case_id == case_uuid)
        .first()
    )
    if row:
        return row

    content = default_factory()
    row = _new_model(
        model,
        {
            "id": uuid.uuid4(),
            "owner_id": owner_uuid,
            "case_id": case_uuid,
            "content_json": content,
            "search_text": renderer(content),
            "version": 1,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
    )
    _write_target_content(target_type, row, content)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_or_create_case_brief(db: Session, owner_id: Any, case_id: Any) -> Any:
    return _get_or_create_target(db, BRIEF_TARGET, owner_id, case_id)


def get_or_create_case_strategy(db: Session, owner_id: Any, case_id: Any) -> Any:
    return _get_or_create_target(db, STRATEGY_TARGET, owner_id, case_id)


def _create_revision(
    db: Session,
    target_type: MemoryTargetType,
    target: Any,
    previous_content: Dict[str, Any],
    new_content: Dict[str, Any],
    source_refs: List[Dict[str, Any]],
    actor: str,
) -> Any:
    revision_model = _model("CaseMemoryRevision")
    revision = _new_model(
        revision_model,
        {
            "id": uuid.uuid4(),
            "owner_id": getattr(target, "owner_id", None),
            "case_id": getattr(target, "case_id", None),
            "target_type": target_type,
            "target_id": getattr(target, "id", None),
            "revision_number": _target_version(db, target_type, target),
            "change_type": "manual" if actor == "user" else "proposal",
            "summary": f"{target_type} updated by {actor}",
            "previous_content_json": previous_content,
            "new_content_json": new_content,
            "before_snapshot": previous_content,
            "after_snapshot": new_content,
            "source_refs": source_refs,
            "actor": actor,
            "created_by": actor,
            "created_at": datetime.utcnow(),
        },
    )
    db.add(revision)
    return revision


def _create_source_records(
    db: Session,
    target_type: MemoryTargetType,
    target: Any,
    source_refs: List[Dict[str, Any]],
) -> None:
    _, source_model, target_column, _, _, _, _ = _target_spec(target_type)
    for source_ref in source_refs:
        source = _new_model(
            source_model,
            {
                "id": uuid.uuid4(),
                "owner_id": getattr(target, "owner_id", None),
                "case_id": getattr(target, "case_id", None),
                target_column: getattr(target, "id", None),
                "source_type": source_ref.get("source_type"),
                "source_id": source_ref.get("source_id"),
                "label": source_ref.get("label", ""),
                "title": source_ref.get("label", ""),
                "excerpt": source_ref.get("excerpt", ""),
                "relevance": source_ref.get("metadata", {}).get("relevance", ""),
                "metadata_": source_ref.get("metadata", {}),
                "metadata_json": source_ref.get("metadata", {}),
                "created_at": datetime.utcnow(),
            },
        )
        source_id = source_ref.get("source_id")
        if source_id and source_ref.get("source_type") == "document":
            try:
                _set_if_column(source, "document_id", _uuid(source_id, "source_id"))
            except ValueError:
                pass
        if source_id and source_ref.get("source_type") in {"research_run", "rechtsprechung_entry"}:
            try:
                _set_if_column(source, "research_source_id", _uuid(source_id, "source_id"))
            except ValueError:
                pass
        db.add(source)


def _update_target_content(
    db: Session,
    target_type: MemoryTargetType,
    target: Any,
    content_json: Dict[str, Any],
    expected_version: Optional[int],
    source_refs: Iterable[Any],
    actor: str,
) -> Any:
    _, _, _, _, renderer, _, _ = _target_spec(target_type)
    if expected_version is not None and _target_version(db, target_type, target) != expected_version:
        raise ValueError("Memory version mismatch")

    refs = _source_refs_to_dicts(source_refs)
    previous_content = _target_content(target_type, target)
    if target_type == BRIEF_TARGET:
        new_content = _validate_brief_content(content_json)
    else:
        new_content = _validate_strategy_content(content_json)

    _create_revision(db, target_type, target, previous_content, new_content, refs, actor)
    if refs:
        _create_source_records(db, target_type, target, refs)

    _write_target_content(target_type, target, new_content)
    _set_if_column(target, "search_text", renderer(new_content))
    if _has_column(target, "version"):
        target.version = int(getattr(target, "version", 0) or 0) + 1
    _set_if_column(target, "updated_at", datetime.utcnow())
    db.add(target)
    db.commit()
    db.refresh(target)
    return target


def update_case_brief_manual(
    db: Session,
    owner_id: Any,
    case_id: Any,
    content_json: Dict[str, Any],
    expected_version: Optional[int] = None,
    source_refs: Optional[Iterable[Any]] = None,
    actor: str = "user",
) -> Any:
    target = get_or_create_case_brief(db, owner_id, case_id)
    return _update_target_content(
        db, BRIEF_TARGET, target, content_json, expected_version, source_refs or [], actor
    )


def update_case_strategy_manual(
    db: Session,
    owner_id: Any,
    case_id: Any,
    content_json: Dict[str, Any],
    expected_version: Optional[int] = None,
    source_refs: Optional[Iterable[Any]] = None,
    actor: str = "user",
) -> Any:
    target = get_or_create_case_strategy(db, owner_id, case_id)
    return _update_target_content(
        db, STRATEGY_TARGET, target, content_json, expected_version, source_refs or [], actor
    )


def _parse_path(path: str) -> List[str]:
    if not path or not path.startswith("/"):
        raise ValueError("Patch path must be an absolute JSON pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in path.split("/")[1:]]


def _apply_patch_ops(
    target_type: MemoryTargetType,
    content: Dict[str, Any],
    ops: Iterable[Any],
) -> Dict[str, Any]:
    _, _, _, _, _, list_fields, scalar_fields = _target_spec(target_type)
    patched = copy.deepcopy(content or {})
    parsed_ops = [_model_dump(op) for op in ops]
    if not parsed_ops:
        raise ValueError("Patch must contain at least one operation")

    for op in parsed_ops:
        operation = op.get("op")
        parts = _parse_path(str(op.get("path") or ""))
        if not parts:
            raise ValueError("Patch path must target a field")
        field = parts[0]
        if field not in list_fields and field not in scalar_fields:
            raise ValueError(f"Patch path is not allowlisted: /{field}")

        if operation == "append":
            if field not in list_fields:
                raise ValueError(f"append is only allowed for list fields: /{field}")
            if len(parts) > 2 or (len(parts) == 2 and parts[1] != "-"):
                raise ValueError("append path must be /field or /field/-")
            patched.setdefault(field, [])
            if not isinstance(patched[field], list):
                raise ValueError(f"Patch target is not a list: /{field}")
            patched[field].append(op.get("value"))
            continue

        if operation == "set":
            if len(parts) == 1:
                patched[field] = op.get("value")
                continue
            if field not in list_fields or len(parts) != 2:
                raise ValueError("set only supports /field or /list_field/index")
            index = int(parts[1])
            patched.setdefault(field, [])
            patched[field][index] = op.get("value")
            continue

        if operation == "remove":
            if field not in list_fields or len(parts) != 2:
                raise ValueError("remove only supports /list_field/index")
            index = int(parts[1])
            patched.setdefault(field, [])
            del patched[field][index]
            continue

        raise ValueError(f"Unsupported patch operation: {operation}")

    if target_type == BRIEF_TARGET:
        return _validate_brief_content(patched)
    return _validate_strategy_content(patched)


def create_memory_update_proposal(
    db: Session,
    owner_id: Any,
    target_type: MemoryTargetType,
    expected_version: int,
    ops: Iterable[MemoryPatchOperation],
    source_refs: Iterable[MemorySourceRef],
    case_id: Optional[Any] = None,
    target_id: Optional[Any] = None,
    confidence: Optional[float] = None,
    model: Optional[str] = None,
) -> Any:
    ops_list = [_model_dump(op) for op in ops]
    refs = _source_refs_to_dicts(source_refs)
    if not refs:
        raise ValueError("source_refs are required")

    if target_id:
        target = _get_target(db, target_type, owner_id, target_id)
    elif case_id:
        target = _get_or_create_target(db, target_type, owner_id, case_id)
    else:
        raise ValueError("case_id or target_id is required")

    content = _target_content(target_type, target)
    _apply_patch_ops(target_type, content, ops_list)
    proposed_patch = {
        "expected_version": expected_version,
        "ops": ops_list,
        "source_refs": refs,
    }
    proposal_model = _model("MemoryUpdateProposal")
    proposal = _new_model(
        proposal_model,
        {
            "id": uuid.uuid4(),
            "owner_id": _uuid(owner_id, "owner_id"),
            "case_id": getattr(target, "case_id", None),
            "target_type": target_type,
            "target_id": getattr(target, "id", None),
            "status": "pending",
            "proposal_type": "patch",
            "title": f"Vorgeschlagene Aktualisierung: {target_type}",
            "rationale": "",
            "expected_version": expected_version,
            "base_version": expected_version,
            "ops": ops_list,
            "source_refs": refs,
            "proposed_patch": proposed_patch,
            "source_payload": {"source_refs": refs},
            "confidence": confidence,
            "model": model,
            "metadata_": {"expected_version": expected_version},
            "metadata_json": {"expected_version": expected_version},
            "created_at": datetime.utcnow(),
        },
    )
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


def list_memory_update_proposals(
    db: Session,
    owner_id: Any,
    case_id: Optional[Any] = None,
    status: Optional[str] = None,
    target_type: Optional[MemoryTargetType] = None,
    limit: int = 50,
) -> List[Any]:
    proposal_model = _model("MemoryUpdateProposal")
    query = db.query(proposal_model).filter(proposal_model.owner_id == _uuid(owner_id, "owner_id"))
    if case_id is not None:
        query = query.filter(proposal_model.case_id == _uuid(case_id, "case_id"))
    if status:
        query = query.filter(proposal_model.status == status)
    if target_type:
        query = query.filter(proposal_model.target_type == target_type)
    return query.order_by(desc(proposal_model.created_at)).limit(limit).all()


def _proposal_expected_version(proposal: Any) -> Optional[int]:
    for attr in ("expected_version", "base_version"):
        value = getattr(proposal, attr, None)
        if value is not None:
            return int(value)
    metadata = _metadata_value(proposal)
    value = metadata.get("expected_version")
    if value is None:
        proposed_patch = getattr(proposal, "proposed_patch", None) or {}
        value = proposed_patch.get("expected_version")
    return int(value) if value is not None else None


def _proposal_ops(proposal: Any) -> List[Dict[str, Any]]:
    ops = getattr(proposal, "ops", None)
    if ops:
        return list(ops)
    proposed_patch = getattr(proposal, "proposed_patch", None) or {}
    return list(proposed_patch.get("ops") or [])


def _proposal_source_refs(proposal: Any) -> List[Dict[str, Any]]:
    refs = getattr(proposal, "source_refs", None)
    if refs:
        return list(refs)
    proposed_patch = getattr(proposal, "proposed_patch", None) or {}
    refs = proposed_patch.get("source_refs")
    if refs:
        return list(refs)
    source_payload = getattr(proposal, "source_payload", None) or {}
    return list(source_payload.get("source_refs") or [])


def accept_memory_update_proposal(
    db: Session,
    owner_id: Any,
    proposal_id: Any,
    actor: str = "user",
) -> Any:
    proposal_model = _model("MemoryUpdateProposal")
    proposal = (
        db.query(proposal_model)
        .filter(
            proposal_model.id == _uuid(proposal_id, "proposal_id"),
            proposal_model.owner_id == _uuid(owner_id, "owner_id"),
        )
        .first()
    )
    if not proposal:
        raise ValueError("Memory update proposal not found")
    if getattr(proposal, "status", None) != "pending":
        raise ValueError("Only pending memory update proposals can be accepted")

    target_type = getattr(proposal, "target_type")
    target = _get_target(db, target_type, owner_id, getattr(proposal, "target_id"))
    expected_version = _proposal_expected_version(proposal)
    if expected_version is None:
        raise ValueError("Proposal is missing expected_version")
    if _target_version(db, target_type, target) != expected_version:
        raise ValueError("Memory version mismatch")

    refs = _proposal_source_refs(proposal)
    if not refs:
        raise ValueError("Proposal source_refs are required")

    previous_content = _target_content(target_type, target)
    new_content = _apply_patch_ops(target_type, previous_content, _proposal_ops(proposal))
    _, _, _, _, renderer, _, _ = _target_spec(target_type)
    _create_revision(db, target_type, target, previous_content, new_content, refs, actor)
    _create_source_records(db, target_type, target, refs)

    _write_target_content(target_type, target, new_content)
    _set_if_column(target, "search_text", renderer(new_content))
    if _has_column(target, "version"):
        target.version = int(getattr(target, "version", 0) or 0) + 1
    _set_if_column(target, "updated_at", datetime.utcnow())

    proposal.status = "accepted"
    _set_if_column(proposal, "reviewed_at", datetime.utcnow())
    _set_if_column(proposal, "reviewed_by", actor)
    _set_if_column(proposal, "updated_at", datetime.utcnow())
    db.add(target)
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


def reject_memory_update_proposal(
    db: Session,
    owner_id: Any,
    proposal_id: Any,
    actor: str = "user",
) -> Any:
    proposal_model = _model("MemoryUpdateProposal")
    proposal = (
        db.query(proposal_model)
        .filter(
            proposal_model.id == _uuid(proposal_id, "proposal_id"),
            proposal_model.owner_id == _uuid(owner_id, "owner_id"),
        )
        .first()
    )
    if not proposal:
        raise ValueError("Memory update proposal not found")
    if getattr(proposal, "status", None) != "pending":
        raise ValueError("Only pending memory update proposals can be rejected")
    proposal.status = "rejected"
    _set_if_column(proposal, "reviewed_at", datetime.utcnow())
    _set_if_column(proposal, "reviewed_by", actor)
    _set_if_column(proposal, "updated_at", datetime.utcnow())
    db.add(proposal)
    db.commit()
    db.refresh(proposal)
    return proposal


def create_case_document_extraction(
    db: Session,
    owner_id: Any,
    case_id: Any,
    document_id: Any,
    extraction_json: Dict[str, Any],
    source_refs: Optional[Iterable[Any]] = None,
    model: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Any:
    extraction_model = _model("CaseDocumentExtraction")
    facts = extraction_json.get("facts") or extraction_json.get("extracted_facts") or []
    entities = extraction_json.get("entities") or extraction_json.get("extracted_entities") or []
    dates = extraction_json.get("dates") or extraction_json.get("extracted_dates") or []
    claims = extraction_json.get("claims") or extraction_json.get("extracted_claims") or []
    extraction = _new_model(
        extraction_model,
        {
            "id": uuid.uuid4(),
            "owner_id": _uuid(owner_id, "owner_id"),
            "case_id": _uuid(case_id, "case_id"),
            "document_id": _uuid(document_id, "document_id"),
            "extraction_type": "memory",
            "status": "completed",
            "extraction_json": extraction_json,
            "extracted_facts": facts,
            "extracted_entities": entities,
            "extracted_dates": dates,
            "extracted_claims": claims,
            "extracted_text_digest": extraction_json.get("digest") or extraction_json.get("summary"),
            "source_refs": _source_refs_to_dicts(source_refs or []),
            "model": model,
            "confidence": confidence,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        },
    )
    db.add(extraction)
    db.commit()
    db.refresh(extraction)
    return extraction


def list_case_document_extractions(
    db: Session,
    owner_id: Any,
    case_id: Any,
    document_id: Optional[Any] = None,
    limit: int = 50,
) -> List[Any]:
    extraction_model = _model("CaseDocumentExtraction")
    query = db.query(extraction_model).filter(
        extraction_model.owner_id == _uuid(owner_id, "owner_id"),
        extraction_model.case_id == _uuid(case_id, "case_id"),
    )
    if document_id is not None:
        query = query.filter(extraction_model.document_id == _uuid(document_id, "document_id"))
    return query.order_by(desc(extraction_model.created_at)).limit(limit).all()


def get_case_memory_prompt_context(
    db: Session,
    current_user: Any,
    case_id: Any,
    include_strategy: bool = True,
    max_chars: int = 5000,
) -> str:
    """Render compact case memory for prompt injection."""
    owner_id = getattr(current_user, "id", current_user)
    chunks: List[str] = []
    try:
        brief = get_or_create_case_brief(db, owner_id, case_id)
        brief_text = render_case_brief_compact(_target_content(BRIEF_TARGET, brief))
        if brief_text and "Keine gepflegten Inhalte" not in brief_text:
            chunks.append(brief_text)
    except Exception as exc:
        print(f"[WARN] Failed to render case brief memory: {exc}")

    if include_strategy:
        try:
            strategy = get_or_create_case_strategy(db, owner_id, case_id)
            strategy_text = render_case_strategy_compact(_target_content(STRATEGY_TARGET, strategy))
            if strategy_text and "Keine gepflegten Inhalte" not in strategy_text:
                chunks.append(strategy_text)
        except Exception as exc:
            print(f"[WARN] Failed to render case strategy memory: {exc}")

    rendered = "\n\n".join(chunks).strip()
    if max_chars and len(rendered) > max_chars:
        return rendered[:max_chars].rstrip() + "\n[Fallgedächtnis gekürzt]"
    return rendered


def memory_row_to_dict(row: Any, rendered: Optional[str] = None) -> Dict[str, Any]:
    target_type = BRIEF_TARGET if type(row).__name__ == "CaseBrief" else STRATEGY_TARGET
    content = _target_content(target_type, row)
    payload = {
        "id": str(getattr(row, "id", "")),
        "owner_id": str(getattr(row, "owner_id", "")),
        "case_id": str(getattr(row, "case_id", "")),
        "content_json": content,
        "search_text": getattr(row, "search_text", "") or "",
        "version": int(getattr(row, "version", 1) or 1),
        "last_reflected_at": None,
        "created_at": None,
        "updated_at": None,
    }
    for key in ("last_reflected_at", "created_at", "updated_at"):
        value = getattr(row, key, None)
        payload[key] = value.isoformat() if value else None
    if rendered is not None:
        payload["rendered"] = rendered
    return payload


def proposal_to_dict(proposal: Any) -> Dict[str, Any]:
    payload = {
        "id": str(getattr(proposal, "id", "")),
        "owner_id": str(getattr(proposal, "owner_id", "")),
        "case_id": str(getattr(proposal, "case_id", "")),
        "target_type": getattr(proposal, "target_type", ""),
        "target_id": str(getattr(proposal, "target_id", "")),
        "status": getattr(proposal, "status", ""),
        "expected_version": _proposal_expected_version(proposal),
        "ops": _proposal_ops(proposal),
        "source_refs": _proposal_source_refs(proposal),
        "confidence": getattr(proposal, "confidence", None),
        "model": getattr(proposal, "model", None),
        "created_at": None,
        "reviewed_at": None,
    }
    for key in ("created_at", "reviewed_at"):
        value = getattr(proposal, key, None)
        payload[key] = value.isoformat() if value else None
    return payload
