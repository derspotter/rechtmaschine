"""
Specialized legal database search functions.
BAMF, RefWorld, EUAA COI, Open Legal Data, EDAL, etc.
"""

import re
import traceback
from typing import Dict, List, Optional

import httpx


async def search_open_legal_data(query: str, court: Optional[str] = None, date_range: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search Open Legal Data API for German case law.
    Args:
        query: Search query string
        court: Optional court filter (e.g., "BGH", "BVerwG")
        date_range: Optional date range (e.g., "2020-2025")
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching Open Legal Data: query='{query}', court={court}, date_range={date_range}")

        # Open Legal Data API appears to be having issues, skip for now
        # The API endpoint may have changed or requires different authentication
        print("Open Legal Data API currently unavailable, skipping")
        return []

    except Exception as e:
        print(f"Error searching Open Legal Data: {e}")
        return []


async def search_refworld(query: str, country: Optional[str] = None, doc_type: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search Refworld (UNHCR) using direct HTTP requests.
    Args:
        query: Search query
        country: Optional country filter
        doc_type: Optional document type filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching Refworld: query='{query}', country={country}, doc_type={doc_type}")

        # Build search URL with query parameters
        search_query = query
        if country:
            search_query += f" {country}"

        search_url = f"https://www.refworld.org/search?query={search_query}&type=caselaw"

        print(f"Fetching Refworld: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            # Refworld typically uses /cases/ or /docid/ URLs
            pattern = r'<a[^>]+href="(/cases/[^"]+|/docid/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                full_url = f"https://www.refworld.org{url_path}"

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "UNHCR Refworld - Rechtsprechung und Länderdokumentation"
                })

            print(f"Refworld returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching Refworld: {e}")
        traceback.print_exc()
        return []


async def search_euaa_coi(query: str, country: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search EUAA COI Portal using direct HTTP requests.
    Args:
        query: Search query
        country: Optional country filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching EUAA COI: query='{query}', country={country}")

        # Build search URL with query parameters
        search_query = query
        if country:
            search_query += f" {country}"

        search_url = f"https://coi.euaa.europa.eu/search?q={search_query}"

        print(f"Fetching EUAA COI: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            pattern = r'<a[^>]+href="(/[^"]*?document[^"]+|/admin/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                full_url = f"https://coi.euaa.europa.eu{url_path}" if not url_path.startswith('http') else url_path

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "EUAA COI Portal - Country of Origin Information"
                })

            print(f"EUAA COI returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching EUAA COI: {e}")
        traceback.print_exc()
        return []


async def search_bamf(query: str, topic: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search BAMF website using direct HTTP requests.
    Args:
        query: Search query
        topic: Optional topic filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching BAMF: query='{query}', topic={topic}")

        # Build search URL
        search_query = query
        if topic:
            search_query += f" {topic}"

        search_url = f"https://www.bamf.de/SiteGlobals/Forms/Suche/EN/Servicessuchformular_formular.html?nn=282388&queryResultId=null&q={search_query}"

        print(f"Fetching BAMF: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            pattern = r'<a[^>]+href="([^"]+)"[^>]*class="[^"]*teaser[^"]*"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                # Make absolute URL
                if url_path.startswith('http'):
                    full_url = url_path
                elif url_path.startswith('/'):
                    full_url = f"https://www.bamf.de{url_path}"
                else:
                    full_url = f"https://www.bamf.de/{url_path}"

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "BAMF - Bundesamt für Migration und Flüchtlinge"
                })

            print(f"BAMF returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching BAMF: {e}")
        traceback.print_exc()
        return []


async def search_edal(query: str, country: Optional[str] = None, court: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Search EDAL (European Database of Asylum Law) using direct HTTP requests.
    Args:
        query: Search query
        country: Optional country filter
        court: Optional court filter
    Returns:
        List of sources with title and url
    """
    try:
        print(f"Searching EDAL: query='{query}', country={country}, court={court}")

        # Build search URL
        search_query = query
        if country:
            search_query += f" {country}"
        if court:
            search_query += f" {court}"

        search_url = f"https://www.asylumlawdatabase.eu/en/content/search?search_api_fulltext={search_query}"

        print(f"Fetching EDAL: {search_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(search_url, follow_redirects=True)
            response.raise_for_status()
            html = response.text

            # Parse HTML to extract results
            sources = []

            # Extract links from search results
            # EDAL uses /en/case-law/ URLs
            pattern = r'<a[^>]+href="(/en/case-law/[^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            for url_path, title in matches[:5]:  # Limit to 5 results
                if not title.strip() or len(title.strip()) < 5:
                    continue

                full_url = f"https://www.asylumlawdatabase.eu{url_path}"

                sources.append({
                    "title": title.strip(),
                    "url": full_url,
                    "description": "EDAL - European Database of Asylum Law"
                })

            print(f"EDAL returned {len(sources)} sources")
            return sources

    except Exception as e:
        print(f"Error searching EDAL: {e}")
        traceback.print_exc()
        return []
