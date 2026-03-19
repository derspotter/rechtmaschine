"""Shared source normalization and ranking utilities for research providers."""

from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


OFFICIAL_SOURCE_HINTS = (
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "justiz.de",
    "berlin.de",
    "justiz-berlin",
    "verwaltungsgericht",
    "bverwg",
    "bverfg",
    "bgh",
    "eur-lex",
    "curia.europa",
    "juris.de",
    "dejure.org",
    "hudoc.echr",
    "ec.europa",
    "ec.eu",
    "openjur.de",
)

OFFTOPIC_HINTS = (
    "blog",
    "nachrichten",
    "news",
    "press",
    "presse",
    "kommentar",
    "comment",
    "anwaltskanzlei",
    "kanzlei",
    "forum",
    "youtube",
    "instagram",
    "facebook",
    "twitter",
    "x.com",
    "linkedin",
)

NON_DECISION_HINTS = (
    "gesetze-im-internet.de",
    "bundesgesetzblatt",
    "grundgesetz",
    "asylvfg_",
    "aufenthg_",
)

DECISION_HINTS = (
    "entscheidung",
    "entscheidungen",
    "entscheid",
    "beschluss",
    "urteilstext",
    "urteil",
    "aktenzeichen",
    "ecli",
    "revision",
)

COURT_HINTS = (
    "bverwg",
    "bverfg",
    "bgh",
    "eugh",
    "egmr",
    "ovg",
    "verwaltungsgericht",
    "vg ",
)

CONTEXT_HINT_TOKENS = (
    "ovg",
    "vg",
    "verwaltungsgericht",
    "bverwg",
    "bverfg",
    "bgh",
    "egmr",
    "eugh",
    "bverf",
)

TRACKING_PARAMS = (
    "utm_",
    "gclid",
    "fbclid",
    "mc_",
    "ref",
    "source",
)

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_RE = re.compile(
    r"\b(?:(?:(?:19|20)\d{2}[./-]\d{1,2}[./-]\d{1,2})|(?:\d{1,2}[./]\d{1,2}[./](?:19|20)\d{2}))\b"
)
ECLI_RE = re.compile(r"\becli:[a-z]{2}:[a-z0-9.\-]+:\d{4}:[\w.\-]+\b", re.IGNORECASE)
CASE_NUMBER_RE = re.compile(
    r"\b(?:az|aktenzeichen)\s*[:\-]?\s*([a-zA-Zäöüß]+\s*[\d./-]+\s*/\s*\d{2,4}|[A-Z]{1,4}\s*[\d./-]+\s*/\s*\d{2,4}|\d{1,4}\s*[A-Za-zÄÖÜäöüß]{0,8}/\d{2,4})\b",
    re.IGNORECASE,
)
PARAGRAPH_RE = re.compile(
    r"(?:§\s*\d+[a-z]?(?:\s*(?:abs\.?|absatz)\s*[a-z0-9]+)?)(?:\s*(?:AsylG|AufenthG|Aufenthg|GG|BVG|AsylVfG|BVerfGG|VwGO|BGB|StGB))?",
    re.IGNORECASE,
)

COURT_PATTERNS = (
    ("BVerwG", ("bverwg",)),
    ("BVerfG", ("bverfg", "bverf")),
    ("EGMR", ("egmr", "european court of human rights", "hudoc.echr")),
    ("EuGH", ("eugh", "curia", "ec.eu")),
    ("BGH", ("bgh",)),
    ("VG", (" verwaltungsgericht", " vg ", "vg_berlin", "verwaltungsgericht berlin")),
    ("OVG", ("ovg", "oberverwaltungsgericht", "vgh")),
    ("VG Berlin", ("vg berlin", "verwaltungsgericht berlin")),
    ("Nrw", ("nrwe.justiz.nrw.de", "justiz.nrw.de")),
)
STOPWORD_RE = re.compile(
    r"\b(?:der|die|das|und|oder|für|von|auf|zu|mit|nach|sowohl|auch|bei|als|gegen|im|in|den|dem|des|ein|eine|einem|einer|nicht|aus|über|vor|hier)\b",
    re.IGNORECASE,
)

BASE_STOPWORDS = {
    "entscheidung", "entscheid", "entscheidungen", "beschluss", "urteil", "aktenzeichen",
    "begründung", "zur", "antrag", "verfahren", "verfahrenes", "verfügung", "recht", "gesetz",
    "vgs", "g",
}


def _blob(source: Dict[str, Any]) -> str:
    return " ".join(
        [
            source.get("url", "") or "",
            source.get("title", "") or "",
            source.get("description", "") or "",
        ]
    ).lower()


def normalize_url(url: str) -> str:
    return (url or "").strip().rstrip(").,;")


def canonical_url(url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""

    try:
        parsed = urlsplit(normalized)
    except Exception:
        return normalized

    if not parsed.netloc:
        return normalized

    filtered_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower()
        if any(lowered == token or lowered.startswith(token) for token in TRACKING_PARAMS):
            continue
        filtered_query.append((key, value))

    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    clean_query = urlencode(filtered_query, doseq=True)
    clean = parsed._replace(netloc=netloc, query=clean_query, fragment="")
    return urlunsplit(clean)


def _title_fingerprint(title: str) -> str:
    cleaned = re.sub(r"[^a-z0-9äöüß]+", " ", (title or "").lower()).strip()
    return re.sub(r"\s+", " ", cleaned)[:160]


def _is_official_source(source: Dict[str, Any]) -> bool:
    text = _blob(source)
    return any(token in text for token in OFFICIAL_SOURCE_HINTS)


def _contains_offtopic_signal(source: Dict[str, Any]) -> bool:
    text = _blob(source)
    return any(token in text for token in OFFTOPIC_HINTS)


def _is_law_text_source(source: Dict[str, Any]) -> bool:
    text = _blob(source)
    return any(token in text for token in NON_DECISION_HINTS)


def _contains_decision_signal(source: Dict[str, Any]) -> int:
    text = _blob(source)
    score = 0
    for token in DECISION_HINTS:
        if token in text:
            score += 10
    if "/entscheid" in text or "/urteil" in text:
        score += 10
    return score


def _contains_court_signal(source: Dict[str, Any]) -> int:
    text = _blob(source)
    score = 0
    for token in COURT_HINTS:
        if token in text:
            score += 8
    return min(score, 24)


def _court_bucket(source: Dict[str, Any]) -> str:
    text = _blob(source)
    if "nrwe.justiz.nrw.de" in text or "justiz.nrw.de" in text:
        return "nrw"
    if "bverwg" in text:
        return "bverwg"
    if "bverfg" in text:
        return "bverfg"
    if "egmr" in text:
        return "egmr"
    if "eugh" in text:
        return "eugh"
    if "bgh" in text:
        return "bgh"
    if "ovg" in text:
        return "ovg"
    if "vg berlin" in text or "verwaltungsgericht berlin" in text:
        return "vg_berlin"
    if "vg " in text or "verwaltungsgericht" in text:
        return "vg"
    return "other"


def _extract_context_terms(context_hints: Optional[List[str]]) -> List[str]:
    if not context_hints:
        return []

    terms: List[str] = []
    seen = set()

    for hint in context_hints:
        normalized = re.sub(r"\s+", " ", str(hint or "").strip()).lower()
        if not normalized:
            continue
        if normalized not in seen:
            terms.append(normalized)
            seen.add(normalized)

        for token in re.findall(r"\b[0-9a-zäöüß./-]+\b", normalized):
            if token in seen:
                continue
            if len(token) >= 4 and (token in CONTEXT_HINT_TOKENS or token.count("/") >= 1):
                terms.append(token)
                seen.add(token)
        for token in CONTEXT_HINT_TOKENS:
            if token in normalized and token not in seen:
                terms.append(token)
                seen.add(token)

    return terms


def _extract_decision_year(source: Dict[str, Any]) -> int:
    text = _blob(source)
    date_years = []
    for match in DATE_RE.finditer(text):
        years = re.findall(r"(?:19|20)\d{2}", match.group(0))
        date_years.extend(int(year) for year in years)
    if date_years:
        return max(date_years)

    years = [int(v) for v in YEAR_RE.findall(text)]
    if not years:
        return 0

    current_year = datetime.utcnow().year
    valid = [year for year in years if year <= current_year + 1]
    return max(valid) if valid else max(years)


def _dedupe_ordered(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if not item:
            continue
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _extract_case_numbers(text: str) -> List[str]:
    normalized = (text or "").lower()
    case_numbers: List[str] = []
    for match in CASE_NUMBER_RE.finditer(normalized):
        matched = match.group(1).replace(" ", "")
        if matched:
            case_numbers.append(matched.upper())

    for value in ECLI_RE.findall(normalized):
        case_numbers.append(value.upper())

    if not case_numbers and ECLI_RE.search(text):
        for value in ECLI_RE.findall(text):
            case_numbers.append(value.upper())

    return _dedupe_ordered(case_numbers)


def _extract_court(text: str) -> Optional[str]:
    normalized = (text or "").lower()
    for court, tokens in COURT_PATTERNS:
        if any(token in normalized for token in tokens):
            return court
    return None


def _extract_paragraphs(text: str) -> List[str]:
    paragraphs = [match.group(0).strip() for match in PARAGRAPH_RE.finditer(text or "")]
    return _dedupe_ordered(paragraphs)


def _extract_keywords(text: str) -> List[str]:
    lowered = (text or "").lower()
    keywords: List[str] = []
    if ECLI_RE.search(lowered):
        keywords.append("ecli")
    if "eu" in lowered and "gericht" in lowered:
        keywords.append("eu")
    if "bverwg" in lowered:
        keywords.append("bverwg")
    if "bverfg" in lowered:
        keywords.append("bverfg")
    if "egmr" in lowered:
        keywords.append("egmr")
    if "eugh" in lowered:
        keywords.append("eugh")
    if "ovg" in lowered:
        keywords.append("ovg")
    if "verwaltungsgericht" in lowered:
        keywords.append("vg")
    if "asyl" in lowered:
        keywords.append("asyl")
    if "dublin" in lowered:
        keywords.append("dublin")

    for token in re.findall(r"\b[a-zäöüß]{4,}\b", lowered):
        if token in BASE_STOPWORDS or STOPWORD_RE.fullmatch(token):
            continue
        if token.isdigit():
            continue
        keywords.append(token)
    return _dedupe_ordered(keywords)[:12]


def normalize_source_entry(
    source: Dict[str, Any],
    provider: str,
    retrieved_at: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    url = normalize_url(source.get("url", "") or "")
    if not url:
        return None

    title = (source.get("title") or "").strip() or url
    description = (
        (source.get("description") or source.get("summary") or "")
        .strip()
    ) or "Relevante Quelle aus Web-Recherche"
    normalized: Dict[str, Any] = {
        "url": url,
        "title": title,
        "description": description,
        "summary": description,
        "source": source.get("source") or provider,
        "provider": provider,
        "document_type": source.get("document_type") or "Rechtsprechung",
        "retrieved_at": retrieved_at or datetime.utcnow().isoformat() + "Z",
    }
    normalized["evidence_type"] = (
        "decision_like" if _contains_decision_signal(normalized) > 0 or _is_official_source(normalized) else "web"
    )

    normalized["pdf_url"] = source.get("pdf_url", None) or normalized.get("pdf_url")
    if source.get("original_url"):
        normalized["original_url"] = source.get("original_url")
    if source.get("resolved_url"):
        normalized["resolved_url"] = source.get("resolved_url")
    if source.get("grounding_segments"):
        normalized["grounding_segments"] = source.get("grounding_segments")
    if source.get("search_queries"):
        normalized["search_queries"] = source.get("search_queries")

    blob = _blob(normalized)
    case_numbers = _extract_case_numbers(blob)
    if case_numbers:
        normalized["case_number"] = case_numbers[0]
        if len(case_numbers) > 1:
            normalized["case_numbers"] = case_numbers
    elif source.get("case_number"):
        normalized["case_number"] = source.get("case_number")
        if source.get("case_numbers"):
            normalized["case_numbers"] = source.get("case_numbers")
    court = _extract_court(blob)
    if court:
        normalized["court"] = court
    elif source.get("court"):
        normalized["court"] = source.get("court")
    publication_year = _extract_decision_year(normalized)
    if publication_year:
        normalized["publication_year"] = publication_year
    elif source.get("publication_year"):
        normalized["publication_year"] = source.get("publication_year")
    paragraphs = _extract_paragraphs(blob)
    if paragraphs:
        normalized["paragraphs"] = paragraphs
    elif source.get("paragraphs"):
        normalized["paragraphs"] = source.get("paragraphs")
    keywords = _extract_keywords(blob)
    if keywords:
        normalized["keywords"] = keywords
    elif source.get("keywords"):
        normalized["keywords"] = source.get("keywords")

    lowered = url.lower()
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        normalized["pdf_url"] = url
    return normalized


def _normalize_source_entry(
    source: Dict[str, Any],
    provider: str,
    retrieved_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_source_entry(
        source,
        provider=provider,
        retrieved_at=retrieved_at,
    )
    if not normalized:
        return None
    return normalized


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for source in sources:
        url_key = canonical_url(source.get("url", ""))
        title_key = _title_fingerprint(source.get("title", ""))
        key = f"{url_key}|{title_key}"
        if not url_key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _score_source(source: Dict[str, Any], context_terms: List[str]) -> Tuple[int, int]:
    return _score_source_with_policy(
        source,
        context_terms=context_terms,
        domain_policy="legal_balanced",
        recency_years=6,
    )


def _score_source_with_policy(
    source: Dict[str, Any],
    *,
    context_terms: List[str],
    domain_policy: str,
    recency_years: int,
) -> Tuple[int, int]:
    score = 0
    text = _blob(source)

    for term in context_terms:
        if term and term in text:
            score += 25

    if _is_official_source(source):
        score += 65

    offtopic_penalty = 80
    law_text_penalty = 220
    if domain_policy == "legal_strict":
        offtopic_penalty = 220
        law_text_penalty = 340
    elif domain_policy == "broad":
        offtopic_penalty = 35
        law_text_penalty = 130

    if _contains_offtopic_signal(source):
        score -= offtopic_penalty
    if _is_law_text_source(source):
        score -= law_text_penalty

    score += _contains_decision_signal(source)
    score += _contains_court_signal(source)

    if ".pdf" in source.get("url", "").lower():
        score += 6

    year = _extract_decision_year(source)
    recency = 0
    if year:
        floor_year = datetime.utcnow().year - max(1, recency_years)
        recency = max(0, year - floor_year)
        if year >= datetime.utcnow().year:
            recency += 8
    elif domain_policy == "legal_strict":
        # Unknown decision date is less desirable in strict legal mode.
        score -= 30

    return score, recency


def _apply_soft_court_diversity(
    scored: List[Tuple[Dict[str, Any], int, int]]
) -> List[Tuple[Dict[str, Any], int, int]]:
    if not scored:
        return []

    has_alternative_courts = any(
        _court_bucket(source) not in ("other", "bverwg")
        for source, _, _ in scored
    )
    preferred_repeats = 1 if has_alternative_courts else 2

    adjusted: List[Tuple[Dict[str, str], int, int]] = []
    court_counts: Dict[str, int] = {}
    for source, score, recency in scored:
        bucket = _court_bucket(source)
        count = court_counts.get(bucket, 0)
        if bucket in ("bverwg", "bverfg") and count >= preferred_repeats:
            score -= 44 * (count - preferred_repeats + 1)
        elif bucket != "other" and count >= 2:
            score -= 28 * (count - 1)
        court_counts[bucket] = count + 1
        adjusted.append((source, score, recency))

    return adjusted


def normalize_and_rank_sources(
    raw_sources: List[Dict[str, Any]],
    *,
    provider: str,
    context_hints: Optional[List[str]] = None,
    limit: int = 40,
    domain_policy: str = "legal_balanced",
    recency_years: int = 6,
    stats: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    input_count = len(raw_sources or [])
    if not raw_sources:
        if stats is not None:
            stats.update(
                {
                    "input_count": input_count,
                    "normalized_count": 0,
                    "deduped_count": 0,
                    "filtered_count": input_count,
                    "reranked_count": 0,
                }
            )
        return []

    retrieved_at = datetime.utcnow().isoformat() + "Z"
    normalized = []
    for source in raw_sources:
        normalized_source = _normalize_source_entry(
            source,
            provider=provider,
            retrieved_at=retrieved_at,
        )
        if normalized_source:
            normalized.append(normalized_source)

    if not normalized:
        if stats is not None:
            stats.update(
                {
                    "input_count": input_count,
                    "normalized_count": 0,
                    "deduped_count": 0,
                    "filtered_count": input_count,
                    "reranked_count": 0,
                }
            )
        return []

    deduped = _dedupe_sources(normalized)
    context_terms = _extract_context_terms(context_hints)

    scored: List[Tuple[Dict[str, Any], int, int]] = []
    for source in deduped:
        score, recency = _score_source_with_policy(
            source,
            context_terms=context_terms,
            domain_policy=domain_policy,
            recency_years=recency_years,
        )
        scored.append((source, score, recency))

    scored.sort(key=lambda item: (item[1], item[2], item[0].get("url", "")), reverse=True)
    diversified = _apply_soft_court_diversity(scored)
    diversified.sort(key=lambda item: (item[1], item[2], item[0].get("url", "")), reverse=True)

    final_sources = [source for source, _, _ in diversified][:limit]
    if stats is not None:
        stats.update(
            {
                "input_count": input_count,
                "normalized_count": len(normalized),
                "deduped_count": len(deduped),
                "filtered_count": max(0, input_count - len(final_sources)),
                "reranked_count": len(final_sources),
            }
        )

    return final_sources
