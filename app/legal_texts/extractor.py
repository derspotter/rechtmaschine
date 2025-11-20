"""
Extract specific legal provisions from downloaded law files.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional
from .downloader import get_law_path


def parse_provision_reference(ref: str) -> Optional[Dict[str, str]]:
    """
    Parse a provision reference string into components.

    Examples:
        "§ 3 AsylG" -> {"law": "AsylG", "paragraph": "3", "absatz": None}
        "§ 60 Abs. 5 AufenthG" -> {"law": "AufenthG", "paragraph": "60", "absatz": "5"}
        "Art. 16a GG" -> {"law": "GG", "paragraph": "16a", "absatz": None}

    Args:
        ref: Reference string like "§ 3 AsylG" or "§ 60 Abs. 5 AufenthG"

    Returns:
        Dict with law, paragraph, absatz keys, or None if parsing failed
    """
    ref = ref.strip()

    # Pattern for § references
    pattern_section = r"§\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*([A-Za-zäöüÄÖÜß]+)"
    # Pattern for Art. references (Grundgesetz)
    pattern_article = r"Art\.\s*(\d+[a-z]?)\s*(?:Abs\.\s*(\d+))?\s*([A-Za-zäöüÄÖÜß]+)"

    for pattern in [pattern_section, pattern_article]:
        match = re.search(pattern, ref, re.IGNORECASE)
        if match:
            paragraph, absatz, law = match.groups()
            return {
                "law": law,
                "paragraph": paragraph,
                "absatz": absatz,
            }

    return None


def extract_provision(
    law: str,
    paragraph: str,
    absatz: Optional[str] = None,
    include_full_section: bool = True
) -> str:
    """
    Extract a specific provision from a law file.

    Args:
        law: Law abbreviation (AsylG, AufenthG, GG)
        paragraph: Paragraph number (e.g., "3", "60", "16a")
        absatz: Specific clause number (e.g., "1", "2"), None for entire paragraph
        include_full_section: Include full paragraph even if absatz specified

    Returns:
        Text of the provision, or error message if not found

    Examples:
        extract_provision("AsylG", "3") -> Full text of § 3
        extract_provision("AufenthG", "60", "5") -> Text of § 60 Abs. 5
    """
    law_path = get_law_path(law)

    if not law_path.exists():
        return f"[FEHLER] Gesetzestext für {law} nicht gefunden. Bitte lade die Gesetze herunter."

    try:
        content = law_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[FEHLER] Konnte {law} nicht lesen: {e}"

    # Find the section heading
    # Markdown format: "# § 3 - Title" or "# § 3" or "## § 3" or "### § 3"
    section_pattern = rf"^#+\s*§\s*{re.escape(paragraph)}(?:\s|$|\s*[-–—])"

    # For Grundgesetz, also try Art/Art. pattern (some headings omit the dot)
    if law == "GG":
        section_pattern = rf"^#+\s*(?:§|Art\.?)\s*{re.escape(paragraph)}(?:\s|$|\s*[-–—])"

    lines = content.split("\n")
    section_start = None
    section_end = None

    # Find start of section
    for i, line in enumerate(lines):
        if re.match(section_pattern, line, re.IGNORECASE):
            section_start = i
            break

    if section_start is None:
        return f"[FEHLER] § {paragraph} nicht gefunden in {law}"

    # Find end of section (next heading of same or higher level)
    heading_level = len(re.match(r"^#+", lines[section_start]).group())
    for i in range(section_start + 1, len(lines)):
        if re.match(rf"^#{{{1,{heading_level}}}}\s", lines[i]):
            section_end = i
            break

    if section_end is None:
        section_end = len(lines)

    # Extract section text
    section_lines = lines[section_start:section_end]
    section_text = "\n".join(section_lines).strip()

    # If specific absatz requested, try to extract it (best effort)
    if absatz and not include_full_section:
        absatz_pattern = rf"\(({absatz})\)\s+([^\(]+?)(?=\(|$)"
        match = re.search(absatz_pattern, section_text, re.DOTALL)
        if match:
            return f"# § {paragraph} Abs. {absatz} {law}\n\n({absatz}) {match.group(2).strip()}"

    # Return full section
    return section_text


def extract_multiple_provisions(provisions: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Extract multiple provisions at once.

    Args:
        provisions: List of dicts with "law", "paragraph", "absatz" keys

    Returns:
        Dict mapping provision reference to extracted text
    """
    results = {}

    for prov in provisions:
        law = prov.get("law", "")
        paragraph = prov.get("paragraph", "")
        absatz = prov.get("absatz")

        ref = f"§ {paragraph} {law}"
        if absatz:
            ref += f" Abs. {absatz}"

        text = extract_provision(law, paragraph, absatz)
        results[ref] = text

    return results


def get_provision_summary(law: str, paragraph: str, max_length: int = 200) -> str:
    """
    Get a short summary/first sentence of a provision.

    Args:
        law: Law abbreviation
        paragraph: Paragraph number
        max_length: Maximum length of summary

    Returns:
        First sentence or truncated text
    """
    full_text = extract_provision(law, paragraph)

    if full_text.startswith("[FEHLER]"):
        return full_text

    # Remove markdown heading
    lines = full_text.split("\n")
    content_lines = [line for line in lines if not line.startswith("#")]
    content = "\n".join(content_lines).strip()

    # Get first sentence or truncate
    sentences = re.split(r'[.!?]\s+', content)
    if sentences:
        summary = sentences[0]
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    return content[:max_length] + "..." if len(content) > max_length else content
