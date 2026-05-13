"""Deterministic page-level citation checks for generated legal drafts."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


VERIFIED_ON_CITED_PAGE = "verified_on_cited_page"
FOUND_ON_DIFFERENT_PAGE = "found_on_different_page"
NOT_FOUND = "not_found"
AMBIGUOUS = "ambiguous"
NO_PAGE_TEXT_AVAILABLE = "no_page_text_available"


INTERNAL_NOTE_RE = re.compile(
    r"\b(Aktennotiz|Kanzleinotiz|Notiz|Gespr[aä]chsnotiz|Besprechungsnotiz|Transkript)\b",
    re.IGNORECASE,
)

CITATION_RE = re.compile(
    r"(?P<anlage>Anlage\s+[A-Z]\s*\d+)\s*,?\s*(?P<anlage_page_label>S\.|Seite|page|p\.)\s*(?P<anlage_page>\d+)"
    r"|(?P<bl>Bl\.?\s*(?P<bl_page>\d+)\s*(?P<bl_suffix>f\.|ff\.)?\s*d\.?\s*A\.?)"
    r"|(?P<page_ref>(?<![A-Za-zÄÖÜäöüß])(?P<page_label>S\.|Seite|page|p\.)\s*(?P<page>\d+)(?P<page_suffix>\s*f\.|\s*ff\.)?)",
    re.IGNORECASE,
)

STOPWORDS = {
    "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am",
    "an", "auch", "auf", "aus", "bei", "beim", "bis", "da", "dadurch",
    "damit", "dann", "das", "dass", "dem", "den", "der", "des", "die", "dies",
    "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch",
    "ein", "eine", "einem", "einen", "einer", "eines", "es", "fuer", "für",
    "gegen", "gewesen", "hat", "hatte", "hatten", "hier", "im", "in", "ist",
    "kein", "keine", "keinem", "keinen", "keiner", "mit", "nach", "nicht",
    "noch", "nur", "oder", "ohne", "sich", "sie", "sind", "sowie", "ueber",
    "über", "und", "unter", "vom", "von", "vor", "war", "waren", "wegen",
    "weil", "wenn", "wie", "wird", "wurde", "wurden", "zu", "zum", "zur",
    "abs", "akte", "akten", "anlage", "blatt", "dokument", "seite", "vgl",
    "siehe", "kläger", "klaeger", "klägerin", "klaegerin", "beklagte",
    "bundesamt", "bescheid", "schreiben", "gericht", "vortrag", "vorgetragen",
}


@dataclass
class DocumentText:
    id: str
    label: str
    category: str
    role: str
    pages: Dict[int, str]
    is_internal_note: bool = False
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class CitationCandidate:
    text: str
    start: int
    end: int
    pages: List[int]
    kind: str
    document_hint: str = ""


def verify_page_citations(
    draft_text: str,
    selected_documents: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Verify page pinpoint citations against only the cited pages first."""
    documents = _load_document_texts(selected_documents)
    checks: List[Dict[str, Any]] = []
    warnings = _internal_note_warnings(draft_text, selected_documents)

    for citation in _find_citations(draft_text):
        sentence = _extract_sentence(draft_text, citation.start, citation.end)
        context = _extract_context(draft_text, citation.start, citation.end)
        claim = _claim_text(sentence, citation.text)
        candidates = _identify_candidate_documents(citation, sentence, context, documents)
        base_payload = {
            "citation": citation.text,
            "claim": claim,
            "sentence": sentence,
            "cited_pages": citation.pages,
            "document_hint": citation.document_hint,
            "kind": citation.kind,
        }

        if not candidates:
            checks.append({
                **base_payload,
                "status": AMBIGUOUS,
                "reason": "Kein passendes Dokument konnte zur Fundstelle bestimmt werden.",
            })
            continue

        if len(candidates) > 1:
            unique_check = _verify_against_candidate_documents(candidates, citation.pages, claim)
            if unique_check:
                checks.append({**base_payload, **unique_check})
            else:
                checks.append({
                    **base_payload,
                    "status": AMBIGUOUS,
                    "candidate_documents": [_document_payload(doc) for doc in candidates],
                    "reason": "Mehrere Dokumente passen zur Fundstelle.",
                })
            continue

        document = candidates[0]
        check = _verify_against_document(document, citation.pages, claim)
        checks.append({
            **base_payload,
            **check,
            "document": _document_payload(document),
        })

    return {
        "checks": checks,
        "warnings": warnings,
        "summary": _summarize_checks(checks),
    }


def _load_document_texts(selected_documents: Dict[str, List[Dict[str, Any]]]) -> List[DocumentText]:
    documents: List[DocumentText] = []
    for category, entries in selected_documents.items():
        for entry in entries or []:
            label = str(entry.get("filename") or entry.get("title") or entry.get("id") or "").strip()
            if not label:
                continue
            page_texts = _page_texts_from_entry(entry)
            page_start, page_end = _extract_page_range_hint(label)
            if page_texts and page_start:
                page_texts = _add_original_page_aliases(page_texts, page_start, page_end)
            documents.append(
                DocumentText(
                    id=str(entry.get("id") or label),
                    label=label,
                    category=category,
                    role=str(entry.get("role") or ""),
                    pages=page_texts,
                    is_internal_note=category == "internal_notes" or bool(INTERNAL_NOTE_RE.search(label)),
                    page_start=page_start,
                    page_end=page_end,
                )
            )
    return documents


def _page_texts_from_entry(entry: Dict[str, Any]) -> Dict[int, str]:
    explicit = entry.get("page_texts") or entry.get("pages")
    if isinstance(explicit, dict):
        return {
            int(page): str(text)
            for page, text in explicit.items()
            if _is_positive_int(page) and str(text).strip()
        }
    if isinstance(explicit, list):
        return {
            idx + 1: str(text)
            for idx, text in enumerate(explicit)
            if str(text).strip()
        }

    for path_key in ("extracted_text_path", "attachment_text_path"):
        path_value = entry.get(path_key)
        if path_value:
            text_path = Path(str(path_value))
            if text_path.exists():
                try:
                    pages = _split_page_text(text_path.read_text(encoding="utf-8"))
                    if pages:
                        return pages
                except Exception as exc:
                    print(f"[WARN] Failed to read page text {text_path}: {exc}")

    file_path = entry.get("file_path") or entry.get("download_path")
    if file_path:
        return _extract_pdf_pages(str(file_path))

    return {}


def _split_page_text(text: str) -> Dict[int, str]:
    if not text.strip():
        return {}

    if "\f" in text:
        return {
            idx + 1: page.strip()
            for idx, page in enumerate(text.split("\f"))
            if page.strip()
        }

    header_re = re.compile(
        r"(?im)^\s*-{2,}\s*(?:Page|Seite)\s+(\d+)\s*-{2,}\s*$"
    )
    matches = list(header_re.finditer(text))
    if matches:
        pages: Dict[int, str] = {}
        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            page_number = int(match.group(1))
            page_text = text[start:end].strip()
            if page_text:
                pages[page_number] = page_text
        return pages

    return {}


def _extract_pdf_pages(file_path: str) -> Dict[int, str]:
    path_obj = Path(file_path)
    if not path_obj.exists() or path_obj.suffix.lower() != ".pdf":
        return {}
    try:
        import fitz

        pages: Dict[int, str] = {}
        with fitz.open(str(path_obj)) as pdf_doc:
            for idx, page in enumerate(pdf_doc):
                text = page.get_text().strip()
                if text:
                    pages[idx + 1] = text
        return pages
    except Exception as exc:
        print(f"[WARN] Failed to extract PDF pages for citation verification {file_path}: {exc}")
        return {}


def _find_citations(text: str) -> List[CitationCandidate]:
    citations: List[CitationCandidate] = []
    for match in CITATION_RE.finditer(text or ""):
        raw = match.group(0).strip()
        if match.group("anlage"):
            pages = [int(match.group("anlage_page"))]
            citations.append(
                CitationCandidate(
                    text=raw,
                    start=match.start(),
                    end=match.end(),
                    pages=pages,
                    kind="anlage_page",
                    document_hint=_normalize_spaces(match.group("anlage")),
                )
            )
            continue

        if match.group("bl"):
            page = int(match.group("bl_page"))
            suffix = (match.group("bl_suffix") or "").lower()
            citations.append(
                CitationCandidate(
                    text=raw,
                    start=match.start(),
                    end=match.end(),
                    pages=_expand_pages(page, suffix),
                    kind="court_file_page",
                    document_hint="akte",
                )
            )
            continue

        page = int(match.group("page"))
        suffix = (match.group("page_suffix") or "").strip().lower()
        if _looks_like_legal_sentence_reference(text, match.start()):
            continue
        citations.append(
            CitationCandidate(
                text=raw,
                start=match.start(),
                end=match.end(),
                pages=_expand_pages(page, suffix),
                kind="page",
            )
        )
    return citations


def _expand_pages(page: int, suffix: str) -> List[int]:
    if suffix == "f.":
        return [page, page + 1]
    if suffix == "ff.":
        return [page, page + 1, page + 2]
    return [page]


def _extract_sentence(text: str, start: int, end: int) -> str:
    left = start
    while left > 0 and not _is_sentence_boundary(text, left - 1):
        left -= 1
    right = end
    while right < len(text) and not _is_sentence_boundary(text, right):
        right += 1
    if right < len(text) and text[right] in ".!?":
        right += 1
    return _normalize_spaces(text[left:right])


def _is_sentence_boundary(text: str, idx: int) -> bool:
    char = text[idx]
    if char == "\n":
        return True
    if char in "!?":
        return True
    if char != ".":
        return False

    previous_char = text[idx - 1] if idx > 0 else ""
    next_char = text[idx + 1] if idx + 1 < len(text) else ""
    if previous_char.isdigit() and next_char.isdigit():
        return False

    prefix = text[max(0, idx - 12):idx + 1]
    if re.search(r"\b(?:z\.B|d\.A|Abs|Nr|S|Bl|vgl|bzw|ca|Dr|Prof)\.$", prefix, re.IGNORECASE):
        return False
    return True


def _extract_context(text: str, start: int, end: int, window: int = 320) -> str:
    return _normalize_spaces(text[max(0, start - window): min(len(text), end + window)])


def _claim_text(sentence: str, citation_text: str) -> str:
    citation_match = re.search(re.escape(citation_text), sentence, flags=re.IGNORECASE)
    if citation_match:
        before = sentence[:citation_match.start()].rstrip()
        if before.endswith("("):
            before = before[:-1].rstrip()
        claim_source = _nearest_supported_clause(before) or sentence
    else:
        claim_source = sentence

    claim = claim_source.replace(citation_text, " ")
    claim = re.sub(r"\([^)]*\)", " ", claim)
    claim = re.sub(r"\b(vgl\.?|siehe|vergleiche|vgl)\b", " ", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\b(Anlage\s+[A-Z]\s*\d+|Bl\.?\s*\d+\s*(?:f\.|ff\.)?\s*d\.?\s*A\.?)\b", " ", claim, flags=re.IGNORECASE)
    claim = re.sub(r"\b(S\.|Seite|page|p\.)\s*\d+\s*(?:f\.|ff\.)?", " ", claim, flags=re.IGNORECASE)
    return _normalize_spaces(claim.strip(" ,;:-"))


def _nearest_supported_clause(text: str) -> str:
    """Return the local clause immediately before a citation when it has enough content."""
    if not text.strip():
        return ""
    parts = re.split(r"[;:]\s+|\s+-\s+", text)
    candidate = parts[-1].strip() if parts else text.strip()

    comma_parts = [part.strip() for part in re.split(r",\s+", candidate) if part.strip()]
    if len(comma_parts) > 1:
        last = comma_parts[-1]
        if (
            len(_content_words(_normalize_for_match(last))) >= 3
            and _looks_like_standalone_clause(last)
        ):
            candidate = last

    if len(_content_words(_normalize_for_match(candidate))) >= 3:
        return candidate
    return text.strip()


def _looks_like_standalone_clause(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return bool(
        re.search(
            r"\b(ist|sind|war|waren|wurde|wurden|hat|haben|hatte|hatten|"
            r"gibt|gab|fehlt|besteht|bestand|enthält|enthaelt|beschreibt|"
            r"berichtet|führt|fuehrt|erklärt|erklaert|bestätigt|bestaetigt|"
            r"erhalten|einzuschüchtern|einzuschuechtern)\b",
            normalized,
        )
    )


def _identify_candidate_documents(
    citation: CitationCandidate,
    sentence: str,
    context: str,
    documents: Sequence[DocumentText],
) -> List[DocumentText]:
    usable_docs = [doc for doc in documents if not doc.is_internal_note]
    sentence_norm = _normalize_for_match(sentence)
    context_norm = _normalize_for_match(context)

    if citation.kind == "anlage_page" and citation.document_hint.lower().replace(" ", "") == "anlagek2":
        primary_bescheide = [
            doc for doc in usable_docs
            if doc.category == "bescheid" and doc.role == "primary"
        ]
        if primary_bescheide:
            return primary_bescheide

    if citation.kind == "court_file_page":
        page_covering_docs = [
            doc for doc in usable_docs
            if _document_covers_any_page(doc, citation.pages)
        ]
        if page_covering_docs:
            return page_covering_docs
        akte_docs = [doc for doc in usable_docs if doc.category == "akte"]
        if len(akte_docs) == 1:
            return akte_docs
        if akte_docs:
            return akte_docs

    named_docs = [
        doc for doc in usable_docs
        if _normalize_for_match(doc.label) and _normalize_for_match(doc.label) in context_norm
    ]
    if named_docs:
        return named_docs

    category_keywords = {
        "bescheid": ("bescheid", "ablehnungsbescheid", "widerspruchsbescheid"),
        "anhoerung": ("anhoerung", "anhörung"),
        "vorinstanz": ("vorinstanz", "urteil", "beschluss"),
        "rechtsprechung": ("entscheidung", "urteil", "beschluss"),
        "sonstiges": ("schreiben", "nachweis", "unterlage"),
    }
    for category, keywords in category_keywords.items():
        if any(_normalize_for_match(keyword) in context_norm for keyword in keywords):
            matching_category = [doc for doc in usable_docs if doc.category == category]
            if len(matching_category) == 1:
                return matching_category

    if len(usable_docs) == 1:
        return usable_docs

    return []


def _looks_like_legal_sentence_reference(text: str, citation_start: int) -> bool:
    """Avoid treating statute Satz references as document page citations."""
    prefix = text[max(0, citation_start - 80):citation_start]
    return bool(re.search(r"(§|Art\.?)", prefix, re.IGNORECASE))


def _verify_against_candidate_documents(
    documents: Sequence[DocumentText],
    cited_pages: Sequence[int],
    claim: str,
) -> Optional[Dict[str, Any]]:
    cited_page_matches: List[Dict[str, Any]] = []
    different_page_matches: List[Dict[str, Any]] = []
    no_text_count = 0

    for document in documents:
        check = _verify_against_document(document, cited_pages, claim)
        payload = {**check, "document": _document_payload(document)}
        if check.get("status") == VERIFIED_ON_CITED_PAGE:
            cited_page_matches.append(payload)
        elif check.get("status") == FOUND_ON_DIFFERENT_PAGE:
            different_page_matches.append(payload)
        elif check.get("status") == NO_PAGE_TEXT_AVAILABLE:
            no_text_count += 1

    if len(cited_page_matches) == 1:
        return cited_page_matches[0]
    if len(cited_page_matches) > 1:
        return None
    if len(different_page_matches) == 1:
        return different_page_matches[0]
    if no_text_count == len(documents):
        return {
            "status": NO_PAGE_TEXT_AVAILABLE,
            "candidate_documents": [_document_payload(doc) for doc in documents],
            "reason": "Für keine der möglichen Quellen liegt seitengetrennter Text vor.",
        }
    return None


def _verify_against_document(document: DocumentText, cited_pages: Sequence[int], claim: str) -> Dict[str, Any]:
    if not document.pages:
        return {
            "status": NO_PAGE_TEXT_AVAILABLE,
            "reason": "Für das Dokument liegt kein seitengetrennter Text vor.",
        }

    missing_pages = [page for page in cited_pages if page not in document.pages]
    cited_text = "\n".join(document.pages.get(page, "") for page in cited_pages).strip()
    if not cited_text:
        return {
            "status": NO_PAGE_TEXT_AVAILABLE,
            "missing_cited_pages": missing_pages,
            "reason": "Für die zitierte Seite liegt kein seitengetrennter Text vor.",
        }

    cited_match = _match_claim(claim, cited_text)
    if cited_match["matched"]:
        return {
            "status": VERIFIED_ON_CITED_PAGE,
            "matched_pages": [page for page in cited_pages if page in document.pages],
            "match": cited_match,
        }

    near_cited_match = _is_near_match(cited_match)
    for page_number, page_text in document.pages.items():
        if page_number in cited_pages:
            continue
        different_match = _match_claim(claim, page_text)
        if different_match["matched"] and _is_strong_match(different_match):
            if near_cited_match:
                return {
                    "status": AMBIGUOUS,
                    "matched_pages": [page_number],
                    "missing_cited_pages": missing_pages,
                    "match": different_match,
                    "cited_page_match": cited_match,
                    "reason": "Fuzzy evidence also exists on the cited page; wrong-page finding is not strict enough.",
                }
            return {
                "status": FOUND_ON_DIFFERENT_PAGE,
                "matched_pages": [page_number],
                "missing_cited_pages": missing_pages,
                "match": different_match,
            }

    if near_cited_match:
        return {
            "status": AMBIGUOUS,
            "missing_cited_pages": missing_pages,
            "cited_page_match": cited_match,
            "reason": "Die zitierte Seite enthält schwache fuzzy Hinweise, aber keinen strengen Treffer.",
        }

    if not _is_reliable_miss(claim, cited_text, cited_match):
        return {
            "status": AMBIGUOUS,
            "missing_cited_pages": missing_pages,
            "cited_page_match": cited_match,
            "reason": "Die zitierte Seite enthält Text, aber der fuzzy Abgleich kann einen Fehlnachweis nicht sicher genug belegen.",
        }

    return {
        "status": NOT_FOUND,
        "missing_cited_pages": missing_pages,
        "reason": "Der Anspruchssatz wurde weder auf der zitierten Seite noch auf anderen Seiten gefunden.",
    }


def _is_strong_match(match: Dict[str, Any]) -> bool:
    if not match.get("matched"):
        return False
    method = match.get("method")
    score = float(match.get("score") or 0.0)
    if method == "normalized_exact":
        return True
    if method == "fuzzy_window":
        return score >= 0.82
    if method == "fuzzy_token_set":
        hard_score = match.get("hard_anchor_score")
        matched_tokens = int(match.get("matched_token_count") or 0)
        if hard_score is not None and float(hard_score) >= 0.8 and matched_tokens >= 2:
            return True
        if score >= 0.98 and matched_tokens >= 3:
            return True
        return score >= 0.68 and matched_tokens >= 4
    return False


def _is_near_match(match: Dict[str, Any]) -> bool:
    method = match.get("method")
    score = float(match.get("score") or 0.0)
    if method in {"normalized_exact", "fuzzy_window"}:
        return score >= 0.34
    if method in {"fuzzy_token_set", "token_overlap"}:
        return score >= 0.2
    return False


def _is_reliable_miss(claim: str, cited_text: str, match: Dict[str, Any]) -> bool:
    """Only emit not_found when deterministic evidence for absence is strong."""
    normalized_claim = _normalize_for_match(claim)
    normalized_page = _normalize_for_match(cited_text)
    if not normalized_claim or not normalized_page:
        return False

    quote_anchors = [
        _normalize_for_match(quote)
        for quote in re.findall(r"[\"„“']([^\"„“']{6,})[\"„“']", claim or "")
    ]
    quote_anchors = [quote for quote in quote_anchors if quote]
    if quote_anchors and not any(quote in normalized_page for quote in quote_anchors):
        return True

    hard_anchors = [_normalize_for_match(anchor) for anchor in _extract_hard_anchors(claim)]
    hard_anchors = [anchor for anchor in hard_anchors if anchor]
    if hard_anchors and not any(anchor in normalized_page for anchor in hard_anchors):
        return True

    method = match.get("method")
    score = float(match.get("score") or 0.0)
    if method == "fuzzy_window" and score >= 0.12:
        return False

    claim_words = sorted(set(_content_words(normalized_claim)))
    if len(claim_words) < 5:
        return False

    page_words = set(normalized_page.split())
    matched_words = [word for word in claim_words if _word_in_page(word, page_words)]
    return len(matched_words) == 0


def _match_claim(claim: str, page_text: str) -> Dict[str, Any]:
    normalized_claim = _normalize_for_match(claim)
    normalized_page = _normalize_for_match(page_text)
    if not normalized_claim or not normalized_page:
        return {"matched": False, "method": "empty"}

    if normalized_claim in normalized_page:
        return {"matched": True, "method": "normalized_exact", "score": 1.0}

    token_fuzzy_match = _match_claim_by_token_fuzzy(claim, normalized_claim, normalized_page)
    if token_fuzzy_match["matched"]:
        return token_fuzzy_match

    claim_words = normalized_claim.split()
    if len(claim_words) < 4:
        return {"matched": False, "method": "too_short"}

    significant_claim_words = {
        word for word in claim_words
        if len(word) >= 4 and word not in STOPWORDS
    }
    page_word_set = set(normalized_page.split())
    if significant_claim_words:
        overlap = significant_claim_words & page_word_set
        overlap_ratio = len(overlap) / len(significant_claim_words)
        if overlap_ratio < 0.35:
            return {
                "matched": False,
                "method": "token_overlap",
                "score": round(overlap_ratio, 4),
            }

    page_words = normalized_page.split()
    window_size = min(max(len(claim_words) + 4, 8), max(len(page_words), 1))
    best_score = 0.0
    best_window = ""
    max_start = max(len(page_words) - window_size + 1, 1)
    stride = max(1, min(8, len(claim_words) // 2))
    starts = range(0, max_start, stride)
    for checked, idx in enumerate(starts):
        if checked >= 300:
            break
        window = " ".join(page_words[idx:idx + window_size])
        window_words = set(window.split())
        if significant_claim_words and len(significant_claim_words & window_words) < 2:
            continue
        score = SequenceMatcher(None, normalized_claim, window).ratio()
        if score > best_score:
            best_score = score
            best_window = window[:240]

    return {
        "matched": best_score >= 0.72,
        "method": "fuzzy_window",
        "score": round(best_score, 4),
        "snippet": best_window,
    }


def _match_claim_by_token_fuzzy(
    raw_claim: str,
    normalized_claim: str,
    normalized_page: str,
) -> Dict[str, Any]:
    claim_words = _content_words(normalized_claim)
    if not claim_words:
        return {"matched": False, "method": "fuzzy_token_set", "score": 0.0}

    page_words = set(normalized_page.split())
    unique_claim_words = sorted(set(claim_words))
    matched_words = [word for word in unique_claim_words if _word_in_page(word, page_words)]
    word_score = len(matched_words) / max(len(unique_claim_words), 1)

    hard_anchors = [_normalize_for_match(anchor) for anchor in _extract_hard_anchors(raw_claim)]
    hard_anchors = [anchor for anchor in hard_anchors if anchor]
    matched_hard = [anchor for anchor in hard_anchors if anchor in normalized_page]
    hard_score = len(matched_hard) / max(len(hard_anchors), 1) if hard_anchors else None

    if hard_anchors:
        matched = (
            hard_score >= 0.67
            and len(matched_words) >= 2
            and word_score >= 0.25
        ) or (
            len(matched_hard) >= 2
            and word_score >= 0.2
        )
    else:
        has_distinctive_long_anchor = any(len(word) >= 12 for word in matched_words)
        matched = (
            len(matched_words) >= 4
            and word_score >= 0.55
        ) or (
            len(matched_words) >= 3
            and len(unique_claim_words) <= 8
            and word_score >= 0.38
            and has_distinctive_long_anchor
        )

    return {
        "matched": matched,
        "method": "fuzzy_token_set",
        "score": round(word_score, 4),
        "hard_anchor_score": None if hard_score is None else round(hard_score, 4),
        "matched_token_count": len(matched_words),
        "token_count": len(unique_claim_words),
        "matched_number_count": len(matched_hard),
        "number_count": len(hard_anchors),
    }


def _content_words(normalized_text: str) -> List[str]:
    return [
        word for word in normalized_text.split()
        if len(word) >= 5 and word not in STOPWORDS
    ]


def _word_in_page(word: str, page_words: set[str]) -> bool:
    if word in page_words:
        return True
    if len(word) < 7:
        return False
    prefix = word[:5]
    for page_word in page_words:
        if len(page_word) < 7:
            continue
        if not page_word.startswith(prefix):
            continue
        if SequenceMatcher(None, word, page_word).ratio() >= 0.78:
            return True
    return False


def _extract_hard_anchors(text: str) -> List[str]:
    anchors: List[str] = []
    anchors.extend(re.findall(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", text or ""))
    anchors.extend(re.findall(r"\b\d{4}\b", text or ""))
    anchors.extend(re.findall(r"\b\d+[.,]\d+\b", text or ""))
    anchors.extend(re.findall(r"\b\d+\b", text or ""))
    deduped: List[str] = []
    for anchor in anchors:
        if anchor not in deduped:
            deduped.append(anchor)
    return deduped


def _internal_note_warnings(
    draft_text: str,
    selected_documents: Dict[str, List[Dict[str, Any]]],
) -> List[str]:
    warnings: List[str] = []
    if INTERNAL_NOTE_RE.search(draft_text or ""):
        warnings.append(
            "Der Entwurf erwähnt interne Notizen wie Aktennotiz, Notiz, Gesprächsnotiz oder Transkript. "
            "Diese nicht als Quelle, Anlage oder Fundstelle zitieren."
        )

    note_entries = list(selected_documents.get("internal_notes", []) or [])
    for entry in note_entries:
        label = str(entry.get("filename") or entry.get("title") or "").strip()
        if label and label in (draft_text or ""):
            warnings.append(
                f"Interne Notiz '{label}' wird namentlich erwähnt. Nicht als Quelle oder Fundstelle zitieren."
            )
    return warnings


def _summarize_checks(checks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        VERIFIED_ON_CITED_PAGE: 0,
        FOUND_ON_DIFFERENT_PAGE: 0,
        NOT_FOUND: 0,
        AMBIGUOUS: 0,
        NO_PAGE_TEXT_AVAILABLE: 0,
    }
    for check in checks:
        status = str(check.get("status") or "")
        if status in summary:
            summary[status] += 1
    return summary


def _document_payload(document: DocumentText) -> Dict[str, Any]:
    return {
        "id": document.id,
        "label": document.label,
        "category": document.category,
        "role": document.role,
        "page_start": document.page_start,
        "page_end": document.page_end,
    }


def _extract_page_range_hint(label: str) -> tuple[Optional[int], Optional[int]]:
    match = re.search(r"(?:^|[_\-\s])p(\d+)(?:-(\d+))?(?=\D*$)", label or "", re.IGNORECASE)
    if not match:
        return None, None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    return start, end


def _add_original_page_aliases(
    pages: Dict[int, str],
    page_start: int,
    page_end: Optional[int],
) -> Dict[int, str]:
    aliased = dict(pages)
    sorted_pages = sorted(pages.items())
    for idx, (_, text) in enumerate(sorted_pages):
        original_page = page_start + idx
        if page_end and original_page > page_end:
            break
        aliased.setdefault(original_page, text)
    return aliased


def _document_covers_any_page(document: DocumentText, pages: Sequence[int]) -> bool:
    if not document.page_start:
        return False
    page_end = document.page_end or document.page_start
    return any(document.page_start <= page <= page_end for page in pages)


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9äöüß]+", " ", normalized)
    return _normalize_spaces(normalized)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False
