"""
Shared utility functions for research modules.
PDF detection, source enrichment, and common helpers.
"""

import asyncio
import re
import traceback
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
from playwright.async_api import async_playwright


ASYL_NET_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _looks_like_pdf(headers) -> bool:
    """Check HTTP headers for indicators that the response is a PDF."""
    if not headers:
        return False
    try:
        content_type = str(headers.get('content-type', '')).lower()
    except Exception:
        content_type = ''
    try:
        content_disposition = str(headers.get('content-disposition', '')).lower()
    except Exception:
        content_disposition = ''
    if 'application/pdf' in content_type:
        return True
    if '.pdf' in content_disposition:
        return True
    return False


def _get_text_from_chat_message(content: Any) -> str:
    """Normalize OpenAI-style message content (str or list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    parts.append(str(text_value))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_title_near_url(text: str, url: str) -> Optional[str]:
    """Try to extract a title from text near the URL."""

    # Look for text in brackets before the URL: [Title](url)
    markdown_pattern = rf'\[([^\]]+)\]\({re.escape(url)}\)'
    match = re.search(markdown_pattern, text)
    if match:
        return match.group(1)

    # Look for text immediately before the URL (within 100 chars)
    url_index = text.find(url)
    if url_index > 0:
        before = text[max(0, url_index-100):url_index].strip()
        # Take the last sentence or phrase
        sentences = before.split('.')
        if sentences and len(sentences[-1].strip()) > 5:
            return sentences[-1].strip()

    return None


def _extract_sources_from_grok_response(text: Any) -> List[Dict[str, str]]:
    """
    Extract source URLs and titles from Grok's response text.
    Grok embeds citations in the text - we need to parse them.
    """
    source_text = _get_text_from_chat_message(text)
    if not source_text:
        return []

    sources: List[Dict[str, str]] = []

    # Look for URL patterns
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\])]+'
    urls = re.findall(url_pattern, source_text)

    # Deduplicate and create source objects
    seen_urls = set()
    for url in urls:
        if url not in seen_urls:
            seen_urls.add(url)
            # Try to extract title from surrounding text
            title = _extract_title_near_url(source_text, url)
            sources.append({
                "url": url,
                "title": title or url,
                "description": "",  # Grok doesn't provide separate snippets
                "source": "Grok"
            })

    print(f"[GROK] Extracted {len(sources)} unique URLs from response")
    return sources


def _convert_grok_citations(citations: Optional[Any]) -> List[Dict[str, str]]:
    """
    Convert Grok's structured citations (AnnotationURLCitation objects) into ResearchResult format.
    """
    if not citations:
        return []

    normalized_sources: List[Dict[str, str]] = []

    def _normalize_entry(entry: Any) -> Optional[Dict[str, str]]:
        # Handle string URLs
        if isinstance(entry, str):
            return {"url": entry, "title": entry, "description": "", "source": "Grok"}

        # Handle dict citations
        if isinstance(entry, dict):
            url = entry.get("url") or entry.get("link") or entry.get("href")
            if not url:
                return None
            title = entry.get("title") or entry.get("name") or url
            description = entry.get("description") or entry.get("snippet") or ""
            return {"url": url, "title": title, "description": description, "source": "Grok"}

        # Handle AnnotationURLCitation objects from xAI SDK
        if hasattr(entry, 'url'):
            url = entry.url
            if not url:
                return None
            title = getattr(entry, 'title', None) or url
            description = getattr(entry, 'description', '') or ""
            return {"url": url, "title": title, "description": description, "source": "Grok"}

        return None

    for item in citations:
        normalized = _normalize_entry(item)
        if normalized:
            normalized_sources.append(normalized)

    if normalized_sources:
        print(f"[GROK] Converted {len(normalized_sources)} citations from structured response")
    return normalized_sources


async def enrich_web_sources_with_pdf(
    sources: List[Dict[str, str]],
    max_checks: int = 10,
    concurrency: int = 3
) -> None:
    """Use Playwright to detect direct PDF links for web research sources."""
    if not sources:
        return

    targets = [src for src in sources if src.get('url')][:max_checks]
    if not targets:
        return

    async def detect_pdf_via_http(url: str) -> Optional[str]:
        """Probe a URL via HTTP to determine if it serves a PDF directly."""
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=10.0,
                headers={"User-Agent": ASYL_NET_USER_AGENT}
            ) as client:
                try:
                    head_response = await client.head(url)
                    if _looks_like_pdf(head_response.headers):
                        return str(head_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410}:
                        print(f"HEAD probe failed for {url}: {err}")

                try:
                    get_response = await client.get(
                        url,
                        headers={"Range": "bytes=0-0"}
                    )
                    if _looks_like_pdf(get_response.headers):
                        return str(get_response.url)
                except httpx.HTTPError as err:
                    status = getattr(getattr(err, "response", None), "status_code", None)
                    if status not in {401, 403, 404, 405, 406, 409, 410}:
                        print(f"GET probe failed for {url}: {err}")
        except Exception as exc:
            print(f"HTTP probing failed for {url}: {exc}")
        return None

    async def process_source(browser, source):
        url = source.get('url')
        if not url:
            return

        lowered = url.lower()
        if lowered.endswith('.pdf') or '.pdf?' in lowered:
            source['pdf_url'] = url
            return

        # Early check for openjur.de pattern: .html -> .ppdf (yes, ppdf!)
        if 'openjur.de' in url and url.endswith('.html'):
            potential_pdf_url = url.replace('.html', '.ppdf')
            print(f"[OPENJUR] Trying PDF pattern: {potential_pdf_url}")
            pdf_exists = await detect_pdf_via_http(potential_pdf_url)
            if pdf_exists:
                print(f"[OPENJUR] Found PDF at: {potential_pdf_url}")
                source['pdf_url'] = potential_pdf_url
                return
            else:
                print(f"[OPENJUR] No PDF found at pattern URL, falling back to page scraping")

        direct_pdf = await detect_pdf_via_http(url)
        if direct_pdf:
            source['pdf_url'] = direct_pdf
            return

        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        try:
            response = await page.goto(url, wait_until='load', timeout=15000)

            if response and _looks_like_pdf(response.headers):
                source['pdf_url'] = response.url or url
                return

            current_url = page.url or url
            # Update the source URL with the resolved URL (after following redirects)
            if current_url != url:
                source['url'] = current_url
                print(f"Resolved URL: {url} -> {current_url}")

            lowered_current = current_url.lower()
            if lowered_current.endswith('.pdf') or '.pdf?' in lowered_current:
                source['pdf_url'] = current_url
                return

            base_url = current_url
            # Fallback: store page title if description is missing
            try:
                if not source.get('description'):
                    page_title = await page.title()
                    if page_title:
                        source['description'] = page_title
            except Exception:
                pass

            # Look for PDF links
            print(f"Searching for PDF links on: {current_url}")

            # Strategy 1: Standard PDF links (ending with .pdf)
            pdf_link = await page.query_selector("a[href$='.pdf'], a[href*='.pdf']")
            if pdf_link:
                href = await pdf_link.get_attribute('href')
                print(f"Found PDF link: {href}")
                if href:
                    href = href.strip()
                    if href.lower().startswith('http'):
                        source['pdf_url'] = href
                    else:
                        source['pdf_url'] = urljoin(base_url, href)
                    print(f"Set PDF URL: {source['pdf_url']}")
                    return

            # Strategy 2: Links with "PDF" in text (e.g., "Download PDF", "Entscheidung als PDF")
            pdf_text_links = await page.query_selector_all("a:has-text('PDF'), a:has-text('pdf')")
            for link in pdf_text_links:
                href = await link.get_attribute('href')
                if href:
                    href = href.strip()
                    # Resolve relative URLs
                    full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                    print(f"Found link with 'PDF' in text: {full_href}")
                    source['pdf_url'] = full_href
                    return

            # Strategy 3: Special patterns for known sites
            # nrwe.justiz.nrw.de uses downloadEntscheidung.php
            if 'nrwe.justiz.nrw.de' in current_url or 'justiz.nrw' in current_url:
                download_link = await page.query_selector("a[href*='downloadEntscheidung.php']")
                if download_link:
                    href = await download_link.get_attribute('href')
                    if href:
                        full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                        print(f"Found NRW Justiz PDF download link: {full_href}")
                        source['pdf_url'] = full_href
                        return

            # Strategy 4: openjur.de - look for NRW Justiz download links
            if 'openjur.de' in current_url:
                # openjur.de pages often link to official court PDFs (e.g., NRW Justiz)
                justiz_link = await page.query_selector("a[href*='justiz.nrw.de'], a[href*='downloadEntscheidung.php']")
                if justiz_link:
                    href = await justiz_link.get_attribute('href')
                    if href:
                        full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                        print(f"Found NRW Justiz link from openjur.de: {full_href}")
                        source['pdf_url'] = full_href
                        return

                # Fallback: try replacing .html with .ppdf (openjur's extension)
                if current_url.endswith('.html'):
                    potential_pdf_url = current_url.replace('.html', '.ppdf')
                    pdf_exists = await detect_pdf_via_http(potential_pdf_url)
                    if pdf_exists:
                        print(f"Found openjur PDF by URL pattern: {potential_pdf_url}")
                        source['pdf_url'] = potential_pdf_url
                        return

            # Strategy 5: Links with download attributes
            download_links = await page.query_selector_all("a[download]")
            for link in download_links:
                href = await link.get_attribute('href')
                if href:
                    full_href = href if href.lower().startswith('http') else urljoin(base_url, href)
                    # Check if it's a PDF
                    if '.pdf' in full_href.lower() or await detect_pdf_via_http(full_href):
                        print(f"Found download link: {full_href}")
                        source['pdf_url'] = full_href
                        return

            # Strategy 6: Click JS-driven download buttons (e.g., BVerwG pdfDownload)
            pdf_button = await page.query_selector("button.pdfDownload, a.pdfDownload, button[data-download]")
            if pdf_button:
                try:
                    print("Attempting JS-triggered PDF download via button")
                    download_future = page.wait_for_event("download", timeout=6000)
                    await pdf_button.click()
                    download = await download_future
                    dl_url = download.url
                    if dl_url:
                        # Keep the resolved page URL for reference
                        source['url'] = current_url
                        source['pdf_url'] = dl_url
                        print(f"Downloaded PDF via button: {dl_url}")
                        return
                except Exception as click_exc:
                    print(f"PDF download button click failed: {click_exc}")

            print(f"No PDF link found with any strategy on {current_url}")

            # Strategy 7: Check for PDF iframes
            iframe_pdf = await page.query_selector("iframe[src$='.pdf'], iframe[src*='.pdf']")
            if iframe_pdf:
                src = await iframe_pdf.get_attribute('src')
                print(f"Found PDF iframe: {src}")
                if src:
                    src = src.strip()
                    if src.lower().startswith('http'):
                        source['pdf_url'] = src
                    else:
                        source['pdf_url'] = urljoin(base_url, src)
                    print(f"Set PDF URL from iframe: {source['pdf_url']}")
            else:
                print(f"No PDF iframe found on {current_url}")
        except Exception as exc:
            print(f"Error detecting PDF link for {url}: {exc}")
        finally:
            try:
                await page.close()
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                sem = asyncio.Semaphore(max(1, concurrency))

                async def run(source):
                    async with sem:
                        await process_source(browser, source)

                await asyncio.gather(*(run(src) for src in targets))
            finally:
                await browser.close()
    except Exception as exc:
        print(f"Playwright enrichment failed: {exc}")
        traceback.print_exc()


async def _enrich_sources_with_pdf_detection(sources: List[Dict[str, str]]) -> None:
    """
    Enrich sources with PDF detection using the existing enrich_web_sources_with_pdf function.
    """
    await enrich_web_sources_with_pdf(sources, max_checks=len(sources), concurrency=3)
