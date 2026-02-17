"""
asyl.net search with legal text extraction integration.

Combines asyl.net database search with AI-powered legal provision extraction.
"""

import asyncio
import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus, urljoin
from datetime import datetime

from google.genai import types
from playwright.async_api import async_playwright

from shared import get_gemini_client, get_document_for_upload
from legal_texts import (
    LegalProvision,
    ProvisionsExtractionResult,
    extract_provision,
    get_law_path
)


# asyl.net configuration
ASYL_NET_BASE_URL = "https://www.asyl.net"
ASYL_NET_SEARCH_PATH = "/recht/entscheidungsdatenbank"
ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ASYL_NET_DECISION_HINTS = (
    "entscheidung",
    "entscheidungsdatum",
    "aktenzeichen",
    "verfügung",
    "beschluss",
    "urteil",
    "bverwg",
    "bverfg",
    "egmr",
    "eugh",
    "ovg",
    "vg ",
    "/entschei",
)

ASYL_NET_OFFICIAL_HINTS = (
    "bverwg",
    "bverfg",
    "eugh",
    "egmr",
    "nrwe.justiz.nrw.de",
    "justiz.nrw.de",
    "verwaltungsgericht",
    "verfassungsgericht",
    "openjur",
    "juris.de",
)

ASYL_NET_NOISE_HINTS = (
    "aktuell",
    "nachrichten",
    "news",
    "kommentar",
    "diskussion",
    "blog",
    "forum",
    "presse",
    "anwaltskanzlei",
)

ASYL_NET_COURT_TOKENS = (
    "ovg",
    "vg",
    "verwaltungsgericht",
    "bverwg",
    "bverfg",
    "eugh",
    "egmr",
)

ASYL_NET_YEAR_RE = re.compile(r"\b(?:(?:19|20)\d{2})\b")


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip(").,;")


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(token in lowered for token in tokens)


def _extract_source_year(source: Dict[str, str]) -> int:
    text = " ".join([source.get("url", ""), source.get("title", ""), source.get("description", "")])
    year_candidates = [int(v) for v in ASYL_NET_YEAR_RE.findall(text)]
    if not year_candidates:
        return 0

    current_year = datetime.utcnow().year
    valid = [y for y in year_candidates if y <= current_year + 1]
    if valid:
        return max(valid)
    return max(year_candidates)


def _is_rechtsprechung_like(url: str, title: str, description: str) -> bool:
    blob = f"{url} {title} {description}".lower()
    if _contains_any(blob, ASYL_NET_DECISION_HINTS):
        return True
    if _contains_any(blob, ASYL_NET_COURT_TOKENS):
        return True
    return any(token in blob for token in ASYL_NET_OFFICIAL_HINTS)


def _score_asylnet_source(
    source: Dict[str, str],
    context_hints: Optional[List[str]] = None,
) -> int:
    url = (source.get("url") or "").lower()
    title = (source.get("title") or "").lower()
    description = (source.get("description") or "").lower()
    blob = f"{url} {title} {description}"
    lowered_blob = blob.lower()

    score = 0
    if _contains_any(blob, ASYL_NET_OFFICIAL_HINTS):
        score += 70

    for token in ASYL_NET_DECISION_HINTS:
        if token in lowered_blob:
            score += 12

    if _contains_any(blob, ASYL_NET_COURT_TOKENS):
        score += 35

    if _contains_any(blob, ASYL_NET_NOISE_HINTS):
        score -= 60

    if _contains_any(blob, ("gesetze-im-internet.de", "bgb", "asylvfg_")):
        score -= 240

    # Prefer explicit context anchors or aktenzeichen hints in asyl decisions
    if context_hints:
        lowered_context = " ".join(context_hints).lower()
        if lowered_context and any(token in lowered_blob for token in lowered_context.split()):
            score += 20

    score += 30 if _is_rechtsprechung_like(url, title, description) else 0
    score += 3 * _extract_source_year(source)

    return score


def _prioritize_asylnet_sources(
    sources: List[Dict[str, str]],
    context_hints: Optional[List[str]] = None,
    limit: int = 25,
) -> List[Dict[str, str]]:
    if not sources:
        return []

    scored = [
        (source, _score_asylnet_source(source, context_hints=context_hints))
        for source in sources
    ]
    scored.sort(key=lambda item: (item[1], (item[0].get("url") or "")), reverse=True)

    ranked: List[Dict[str, str]] = []
    decision_urls = set()
    for source, _ in scored:
        url = _normalize_url(source.get("url", ""))
        if not url or url in decision_urls:
            continue
        if _is_rechtsprechung_like(url, source.get("title") or "", source.get("description") or ""):
            decision_urls.add(url)
            ranked.append(source)

    if not ranked:
        ranked = [source for source, _ in scored if _normalize_url(source.get("url", "")) and _normalize_url(source.get("url", "")) not in decision_urls]

    # Keep one source per URL.
    deduped: List[Dict[str, str]] = []
    seen_urls = set()
    for source in ranked:
        url = _normalize_url(source.get("url", ""))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        source["url"] = url
        deduped.append(source)
        if len(deduped) >= limit:
            break

    return deduped


def _build_search_candidates(
    query: str,
    suggestions: Optional[List[str]] = None,
    fallback_suggestions: Optional[List[str]] = None,
) -> List[str]:
    clean_query = (query or "").replace('"', '').replace("'", "").strip()
    first_term = clean_query.split()[0] if clean_query else ""

    normalized_set = set()
    candidates: List[str] = []

    if suggestions:
        prepared = [kw.strip() for kw in suggestions if isinstance(kw, str) and kw.strip()]
        unique_prepared = []
        for kw in prepared:
            low = kw.lower()
            if low not in normalized_set:
                normalized_set.add(low)
                unique_prepared.append(kw)

        if unique_prepared:
            candidates.append(",".join(unique_prepared[:2]))
            candidates.extend(unique_prepared[:3])
            if len(unique_prepared) > 1:
                candidates.append(" ".join(unique_prepared[:2]))

    if clean_query:
        candidates.append(clean_query)

    if not candidates and fallback_suggestions and len(first_term) >= 2:
        for kw in fallback_suggestions:
            if kw.lower() not in normalized_set:
                normalized_set.add(kw.lower())
                candidates.append(kw)

    if not candidates and len(first_term) >= 2:
        fallback = []
        fallback_cache = []
        # Keep sync path non-async safe: only include first term as a fallback candidate
        fallback_cache.append(first_term)
        for kw in fallback_cache:
            if kw.lower() not in normalized_set:
                normalized_set.add(kw.lower())
                fallback.append(kw)
        candidates.extend(fallback)

    # Best-effort uniqueness + deterministic order
    ordered: List[str] = []
    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(candidate.strip())

    return ordered[:3]


# Load cached asyl.net suggestions
ASYL_NET_SUGGESTIONS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "asyl_net_suggestions.json"
try:
    with open(ASYL_NET_SUGGESTIONS_FILE, "r", encoding="utf-8") as f:
        _asyl_suggestions_payload = json.load(f)
    ASYL_NET_ALL_SUGGESTIONS: List[str] = _asyl_suggestions_payload.get("suggestions", [])
except FileNotFoundError:
    print(f"Warning: asyl.net suggestions file not found at {ASYL_NET_SUGGESTIONS_FILE}")
    ASYL_NET_ALL_SUGGESTIONS = []


async def extract_keywords_and_provisions(
    query: str,
    attachment_label: Optional[str] = None,
    attachment_doc: Optional[Dict[str, Optional[str]]] = None,
    existing_upload: Optional[Any] = None,
    client: Optional[Any] = None,
) -> ProvisionsExtractionResult:
    """
    Extract both asyl.net keywords and relevant legal provisions from query/document.

    Uses Gemini with structured output to analyze the query or attached document
    and extract:
    1. asyl.net search keywords (from cached list)
    2. Relevant legal provisions (§§ from AsylG, AufenthG, GG)

    Args:
        query: User's research query
        attachment_label: Optional filename of attached document
        attachment_doc: Optional document metadata dict
        existing_upload: Optional pre-uploaded Gemini file
        client: Optional Gemini client (creates new if not provided)

    Returns:
        ProvisionsExtractionResult with keywords and provisions
    """
    gemini_client = client
    uploaded_file = existing_upload
    uploaded_name: Optional[str] = None
    temp_path: Optional[str] = None

    try:
        print("[PROVISION EXTRACTION] Starting keyword and provision extraction...")
        if gemini_client is None:
            gemini_client = get_gemini_client()

        suggestion_text = "\n".join(f"- {s}" for s in ASYL_NET_ALL_SUGGESTIONS) if ASYL_NET_ALL_SUGGESTIONS else "- (keine Schlagwörter geladen)"
        trimmed_query = (query or "").strip()

        if attachment_label:
            task_description = f"""Analysiere den beigefügten BAMF-Bescheid "{attachment_label}" und leite daraus ab:
1. Die geeignetsten asyl.net Schlagwörter aus der unten stehenden Liste
2. Die relevanten Rechtsgrundlagen (Paragraphen aus AsylG, AufenthG, GG)

Nutze insbesondere Tatbestand, rechtliche Würdigung, Länder- oder Herkunftsbezüge sowie angewandte Rechtsnormen.

Zusätzliche Aufgabenstellung / Notiz:
{trimmed_query or "- (keine zusätzliche Notiz)"}"""
        else:
            task_description = f"""Analysiere die folgende Anfrage und leite daraus ab:
1. Die geeignetsten asyl.net Schlagwörter aus der unten stehenden Liste
2. Die relevanten Rechtsgrundlagen (Paragraphen aus AsylG, AufenthG, GG)

Anfrage:
{trimmed_query or "(Keine Anfrage angegeben)"}"""

        prompt = f"""Du bist ein Rechercheassistent für deutsches Asylrecht.

{task_description}

Hier ist die Schlagwort-Liste für asyl.net (verwende ausschließlich Begriffe aus dieser Liste):
{suggestion_text}

WICHTIGE HINWEISE FÜR SCHLAGWÖRTER:
- Gib GENAU 2 Schlagwörter zurück.
- Schwerpunkt: 1 Begriff zum Herkunftsland/Region des Falls UND 1 Begriff, der den konkreten Fall charakterisiert (z.B. Volkszugehörigkeit, vulnerabler Status, Verfahrensart wie Dublin-III VO).
- Bevorzuge IMMER aktuelle Rechtsbegriffe gegenüber veralteten Begriffen.
- Bei Dublin-Fällen: Nutze "Dublin-III VO" (aktuelle Verordnung), NICHT "Dublin-Übereinkommen" (veraltet).
- Nutze ausschließlich Begriffe aus der obigen Liste.

RECHTSGRUNDLAGEN - Beispiele:
- Flüchtlingsanerkennung → § 3 AsylG (ggf. mit spezifischen Absätzen wie § 3 Abs. 1)
- Subsidiärer Schutz → § 4 AsylG
- Abschiebungsverbot (Syrien, Afghanistan, etc.) → § 60 Abs. 5, 7 AufenthG
- Asylgrundrecht → Art. 16a GG
- Dublin-Verfahren → § 29 AsylG

Gib mir 1-3 Schlagwörter und alle relevanten Rechtsgrundlagen zurück.

Antwortformat (JSON):
{{
  "keywords": ["Schlagwort 1", "Schlagwort 2"],
  "provisions": [
    {{
      "law": "AsylG",
      "paragraph": "3",
      "absatz": ["1"],
      "reasoning": "Flüchtlingsdefinition wird im Bescheid angewendet"
    }},
    {{
      "law": "AufenthG",
      "paragraph": "60",
      "absatz": ["5", "7"],
      "reasoning": "Abschiebungsverbot wegen Gefahrenlage"
    }}
  ]
}}

Keine zusätzlichen Erklärungen, kein Markdown, nur das JSON-Objekt."""

        contents: List[Any] = [prompt]

        # Upload document if needed
        if uploaded_file is None and attachment_doc:
            try:
                upload_entry = dict(attachment_doc)
                file_path, mime_type, needs_cleanup = get_document_for_upload(upload_entry)
                if needs_cleanup:
                    temp_path = file_path
                with open(file_path, "rb") as file_handle:
                    display_name = attachment_label or upload_entry.get("filename") or "Bescheid"
                    uploaded_file = gemini_client.files.upload(
                        file=file_handle,
                        config={
                            "mime_type": mime_type,
                            "display_name": f"{display_name}{'.txt' if mime_type == 'text/plain' else ''}"
                        }
                    )
                uploaded_name = uploaded_file.name
                print(f"[PROVISION EXTRACTION] Uploaded document for analysis: {display_name}")
            except Exception as prep_exc:
                print(f"[PROVISION EXTRACTION] Failed to prepare attachment: {prep_exc}")
                uploaded_file = None

        if uploaded_file:
            contents.append(uploaded_file)

        # Call Gemini with structured output
        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-3-flash-preview",
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0)
        )

        raw_text = response.text if response.text else "{}"

        # Parse JSON response
        try:
            # Clean up markdown code blocks if present
            raw_text = re.sub(r'```json\s*', '', raw_text)
            raw_text = re.sub(r'```\s*$', '', raw_text)

            data = json.loads(raw_text)

            # Extract keywords
            keywords = data.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            keywords = [s.strip() for s in keywords if isinstance(s, str) and s.strip()]

            # Deduplicate keywords
            seen = set()
            unique_keywords = []
            for k in keywords:
                low = k.lower()
                if low not in seen:
                    seen.add(low)
                    unique_keywords.append(k)

            # Extract provisions
            provisions_data = data.get("provisions", [])
            provisions = []
            for prov_dict in provisions_data:
                try:
                    # Validate provision data
                    if not isinstance(prov_dict, dict):
                        continue

                    law = prov_dict.get("law")
                    paragraph = prov_dict.get("paragraph")
                    reasoning = prov_dict.get("reasoning", "")

                    if not law or not paragraph:
                        continue

                    # Handle absatz field
                    absatz = prov_dict.get("absatz")
                    if absatz is not None:
                        if isinstance(absatz, str):
                            absatz = [absatz]
                        elif not isinstance(absatz, list):
                            absatz = None

                    provision = LegalProvision(
                        law=str(law),
                        paragraph=str(paragraph),
                        absatz=absatz,
                        reasoning=str(reasoning)
                    )
                    provisions.append(provision)
                    print(f"[PROVISION EXTRACTION] Extracted: {provision}")

                except Exception as prov_exc:
                    print(f"[PROVISION EXTRACTION] Failed to parse provision: {prov_exc}")
                    continue

            result = ProvisionsExtractionResult(
                keywords=unique_keywords[:2],
                provisions=provisions
            )

            print(f"[PROVISION EXTRACTION] Extracted {len(result.keywords)} keywords and {len(result.provisions)} provisions")
            return result

        except json.JSONDecodeError as json_exc:
            print(f"[PROVISION EXTRACTION] JSON parse error: {json_exc}")
            print(f"[PROVISION EXTRACTION] Raw text: {raw_text[:200]}")
            return ProvisionsExtractionResult(keywords=[], provisions=[])

    except Exception as exc:
        print(f"[PROVISION EXTRACTION] Failed: {exc}")
        traceback.print_exc()
        return ProvisionsExtractionResult(keywords=[], provisions=[])

    finally:
        # Clean up temp file if created
        if temp_path:
            try:
                import os
                os.unlink(temp_path)
            except:
                pass


async def load_legal_provisions(provisions: List[LegalProvision]) -> List[Dict[str, str]]:
    """
    Load legal provision texts from local files and format as sources.

    Args:
        provisions: List of LegalProvision objects to load

    Returns:
        List of source dicts formatted for ResearchResult
    """
    sources = []

    for prov in provisions:
        try:
            # Check if law file exists
            law_path = get_law_path(prov.law)
            if not law_path.exists():
                print(f"[LEGAL TEXTS] Law file not found for {prov.law}, skipping")
                continue

            # Extract provision text
            absatz_str = prov.absatz[0] if prov.absatz and len(prov.absatz) == 1 else None
            text = extract_provision(prov.law, prov.paragraph, absatz_str)

            if text.startswith("[FEHLER]"):
                print(f"[LEGAL TEXTS] Failed to extract {prov}: {text}")
                continue

            # Build title
            title = f"§ {prov.paragraph} {prov.law}"
            if prov.absatz:
                title += f" Abs. {', '.join(prov.absatz)}"

            # Build official URL with paragraph-specific link
            law_bases = {
                "AsylG": "https://www.gesetze-im-internet.de/asylvfg_1992/",
                "AufenthG": "https://www.gesetze-im-internet.de/aufenthg_2004/",
                "GG": "https://www.gesetze-im-internet.de/gg/",
                "AsylbLG": "https://www.gesetze-im-internet.de/asylblg/",
            }
            base_url = law_bases.get(prov.law, "https://www.gesetze-im-internet.de/")

            # Construct paragraph-specific URL (e.g., __3.html for § 3)
            url = f"{base_url}__{prov.paragraph}.html"

            sources.append({
                "title": title,
                "url": url,
                "description": f"📜 {prov.reasoning}",
                "document_type": "Gesetzestext",
                "full_text": text,  # Full provision text for display/saving
                "pdf_url": "",  # No PDF for legal texts (stored as markdown)
                "source": "Gesetzestext",
            })

            print(f"[LEGAL TEXTS] Loaded: {title}")

        except Exception as exc:
            print(f"[LEGAL TEXTS] Failed to load {prov}: {exc}")
            continue

    return sources


async def get_asyl_net_keyword_suggestions(partial_query: str) -> List[str]:
    """
    Get keyword suggestions from cached asyl.net keywords.

    Args:
        partial_query: Partial keyword to match

    Returns:
        List of matching keywords (up to 10)
    """
    try:
        print(f"Getting asyl.net keyword suggestions for: '{partial_query}'")
        if not ASYL_NET_ALL_SUGGESTIONS:
            print("No cached asyl.net suggestions loaded")
            return []

        prefix = (partial_query or "").strip().lower()
        if not prefix:
            return ASYL_NET_ALL_SUGGESTIONS[:10]

        # Prefix matches first
        matches = [s for s in ASYL_NET_ALL_SUGGESTIONS if s.lower().startswith(prefix)]

        # Then substring matches
        if len(matches) < 5:
            matches.extend(
                s for s in ASYL_NET_ALL_SUGGESTIONS
                if prefix in s.lower() and s not in matches
            )

        result = matches[:10]
        print(f"Found {len(result)} keyword suggestions")
        return result

    except Exception as e:
        print(f"Error getting keyword suggestions from cache: {e}")
        traceback.print_exc()
        return []


async def search_asyl_net(
    query: str,
    category: Optional[str] = None,
    suggestions: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    Search asyl.net Rechtsprechungs-Datenbank using cached Schlagwörter.

    Args:
        query: Search query
        category: Optional category filter (e.g., "Dublin", "EGMR")
        suggestions: Schlagwörter to combine in the search

    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching asyl.net: query='{query}', category={category}, suggestions={suggestions}")

        clean_query = query.replace('"', '').replace("'", "").strip()
        fallback = []
        first_term = clean_query.split()[0] if clean_query else ""
        if first_term:
            fallback = await get_asyl_net_keyword_suggestions(first_term)
            fallback = fallback[:3]

        candidate_keywords = _build_search_candidates(query, suggestions, fallback_suggestions=fallback)

        if not candidate_keywords:
            print("asyl.net: no candidate keywords generated")
            return []

        results: List[Dict[str, str]] = []
        seen_urls = set()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                for idx, keyword in enumerate(candidate_keywords):
                    encoded_keywords = quote_plus(keyword)
                    search_url = (
                        f"{ASYL_NET_BASE_URL}{ASYL_NET_SEARCH_PATH}"
                        f"?newsearch=1&keywords={encoded_keywords}&keywordConjunction=1&limit=25"
                    )

                    print(f"Fetching asyl.net results for keyword '{keyword}' (rank {idx})")
                    await page.goto(search_url, wait_until="networkidle", timeout=30000)

                    # Dismiss cookie banner if present
                    try:
                        cookie_button = await page.query_selector(
                            "button#CybotCookiebotDialogBodyLevelButtonAccept, button:has-text('Akzeptieren')"
                        )
                        if cookie_button:
                            await cookie_button.click()
                            await page.wait_for_timeout(500)
                    except Exception:
                        pass

                    items = await page.query_selector_all("div.rsdb_listitem")

                    for item in items[:5]:
                        link_elem = await item.query_selector("div.rsdb_listitem_court a")
                        if not link_elem:
                            continue

                        url = await link_elem.get_attribute("href")
                        if not url:
                            continue

                        if not url.startswith("http"):
                            url = f"{ASYL_NET_BASE_URL}{url}"

                        if url in seen_urls:
                            continue

                        seen_urls.add(url)

                        # Try to find PDF on detail page
                        pdf_url = None
                        detail_page = None
                        try:
                            detail_page = await browser.new_page()
                            await detail_page.goto(url, wait_until="domcontentloaded", timeout=30000)
                            pdf_link_elem = await detail_page.query_selector("a[href$='.pdf'], a[href*='.pdf']")
                            if pdf_link_elem:
                                href = await pdf_link_elem.get_attribute("href")
                                if href:
                                    pdf_url = urljoin(url, href)
                                    print(f"Found PDF link for asyl.net result using '{keyword}': {pdf_url}")
                        except Exception as detail_error:
                            print(f"Could not inspect detail page for {url}: {detail_error}")
                        finally:
                            if detail_page:
                                await detail_page.close()

                        # Extract metadata
                        court_elem = await item.query_selector(".rsdb_listitem_court .courttitle")
                        headnote_elem = await item.query_selector(".rsdb_listitem_court .headnote")
                        footer_elem = await item.query_selector(".rsdb_listitem_footer")

                        def clean_text(text: Optional[str]) -> str:
                            if not text:
                                return ""
                            return re.sub(r"\s+", " ", text).strip()

                        court_text = clean_text(await court_elem.text_content() if court_elem else "")
                        headnote_text = clean_text(await headnote_elem.text_content() if headnote_elem else "")
                        footer_text = clean_text(await footer_elem.text_content() if footer_elem else "")

                        title_parts = [part for part in [court_text, headnote_text] if part]
                        raw_title = " – ".join(title_parts) if title_parts else f"asyl.net Ergebnis zu {keyword}"
                        description_parts = [headnote_text, footer_text]
                        description = " ".join(part for part in description_parts if part)
                        if not description:
                            description = "Rechtsprechungsfundstelle aus der asyl.net Entscheidungsdatenbank."

                        results.append({
                            "title": raw_title,
                            "url": url,
                            "description": description,
                            "pdf_url": pdf_url,
                            "search_keyword": keyword,
                            "document_type": "Rechtsprechung",
                            "source": "asyl.net",
                            "suggestions": ",".join(suggestions) if suggestions else keyword,
                        })

                if results:
                    results = _prioritize_asylnet_sources(results, context_hints=[query], limit=20)
                    print(f"asyl.net returned {len(results)} direct results across {len(candidate_keywords)} keyword variants (after ranking: {len(results)})")
                else:
                    print("asyl.net returned no direct results")

                return results

            finally:
                await browser.close()

    except Exception as e:
        print(f"Error searching asyl.net: {e}")
        traceback.print_exc()
        return []


# Main entry point that combines everything
# Main entry point that combines everything
async def search_asylnet_with_provisions(
    query: str,
    attachment_label: Optional[str] = None,
    attachment_doc: Optional[Dict[str, Optional[str]]] = None,
    existing_upload: Optional[Any] = None,
    client: Optional[Any] = None,
    manual_keywords: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combined asyl.net search with legal provision extraction.

    This is the main entry point that:
    1. Extracts keywords and provisions using AI (unless manual_keywords provided)
    2. Searches asyl.net database
    3. Loads legal provision texts
    4. Returns combined results

    Args:
        query: User query
        attachment_label: Label for the attached file
        attachment_doc: Document metadata
        existing_upload: Existing file upload object
        client: Gemini client
        manual_keywords: Optional manual keywords provided by user (overrides extraction)

    Returns:
        Dict with:
        - keywords: List[str] - asyl.net keywords for suggestions
        - asylnet_sources: List[Dict] - Results from asyl.net
        - legal_sources: List[Dict] - Legal provision texts
    """
    
    # 1. Extract keywords and provisions
    # If manual_keywords are provided, we pass them as a hint or override?
    # Actually, if manual keywords are provided, we should probably still run extraction for provisions
    # but use the manual keywords for the asyl.net search.
    
    extraction_result = await extract_keywords_and_provisions(
        query,
        attachment_label=attachment_label,
        attachment_doc=attachment_doc,
        existing_upload=existing_upload,
        client=client
    )

    # Determine keywords for search
    search_keywords = extraction_result.keywords
    if manual_keywords and manual_keywords.strip():
        # Split by comma if multiple
        manual_list = [k.strip() for k in manual_keywords.split(",") if k.strip()]
        if manual_list:
            print(f"[ASYL.NET] Using manual keywords: {manual_list}")
            search_keywords = manual_list
            # Also update the result keywords so UI knows what was used
            extraction_result.keywords = manual_list

    # 2. Search asyl.net with selected keywords
    asylnet_sources = await search_asyl_net(query, suggestions=search_keywords)

    # 3. Load legal provisions from local files
    legal_sources = await load_legal_provisions(extraction_result.provisions)

    return {
        "keywords": extraction_result.keywords,
        "asylnet_sources": asylnet_sources,
        "legal_sources": legal_sources,
    }
