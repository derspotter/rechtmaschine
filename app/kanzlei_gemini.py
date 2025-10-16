#!/usr/bin/env python3
"""
Extract Anhörung (hearing) and Bescheid (decision) sections from PDF documents
using Google's Gemini 2.5 Flash Preview model.
"""

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pikepdf
import pydantic
from dotenv import load_dotenv
from google import genai
from google.genai import types
import shutil

# Load environment variables
load_dotenv()


class PageRange(pydantic.BaseModel):
    """A range of pages identified in the document."""

    start_page: int = pydantic.Field(
        description="1-based physical page index where section starts"
    )
    end_page: int = pydantic.Field(
        description="1-based physical page index where section ends"
    )
    document_type: str = pydantic.Field(
        description="Type of document: 'Anhörung' or 'Bescheid'"
    )
    confidence: float = pydantic.Field(
        ge=0.0, le=1.0, description="Confidence in identification"
    )
    partial_from_previous: bool = pydantic.Field(
        default=False,
        description="True if the section appears to begin before this part of the document",
    )
    partial_into_next: bool = pydantic.Field(
        default=False,
        description="True if the section appears to continue after this part of the document",
    )


class DocumentSections(pydantic.BaseModel):
    """Identified sections in the legal document."""

    sections: List[PageRange] = pydantic.Field(
        default_factory=list,
        description="List of identified Anhörung and Bescheid sections",
    )


@dataclass(frozen=True)
class GeminiConfig:
    model: str = "gemini-2.5-flash-preview-09-2025"
    temperature: float = 0.0


PROMPT_TEMPLATE = """Analysiere dieses deutsche BAMF-Verwaltungsdokument mit insgesamt {total_pages} physisch nummerierten Seiten (1-basiert). Identifiziere alle vollständigen Abschnitte, die eindeutig zu folgenden Kategorien gehören:

1. **Anhörung** (Anhörungsprotokoll / Niederschrift)
2. **Bescheid** (BAMF-Bescheid / Verwaltungsentscheidung)

**STRIKTE ANFORDERUNGEN**

- **Anhörung (Niederschrift):**
  - Erkennbarer BAMF-Briefkopf mit Bundesadler und „Bundesamt für Migration und Flüchtlinge“.
  - Rechts oben eine Box „Bearbeitende Stelle“ (Referat, Anschriften, Telefon/Fax) sowie ein „Az:“ (Aktenzeichen) in der Nähe der Kopfzeile.
  - Titel beginnt mit „Niederschrift …“ (z. B. „Niederschrift über die Anhörung …“ oder „Niederschrift über die Befragung …“).
  - Der Text beginnt mit Formulierungen wie „Es erscheint der/die Antragsteller/in …“, beschreibt Dolmetscher, Sprache, Belehrungen und besitzt einen Frage/Antwort- oder Fließtext-Dialog zur Anhörung.
  - Am Ende befinden sich Unterschrifts- bzw. Bestätigungsfelder.
  - **Ausschluss:** Einladungen, Kontrollbögen, Checklisten, bloße Erwähnungen oder Dokumente ohne vollständigen Anhörungs-Textkörper.

- **Bescheid (BAMF-Verfügung):**
  - Auffällige, zentrierte Überschrift „BESCHEID“ (häufig als „B E S C H E I D“) mit BAMF-Briefkopf und Bundestitel, sowie Angaben rechts oben (Ort, Datum, „Gesch.-Z.“).
  - Abschnitt mit Personalien des Antragstellers (Name, Geburtsdatum, Adresse, AZR-Nummer) gefolgt von „ergeht folgende Entscheidung:“ und nummerierter Entscheidungsliste.
  - Verweise auf Rechtsgrundlagen (§§ AsylG/AufenthG), Rechtsbehelfsbelehrung und amtliche Fuß- oder Kopfzeilen.
  - **Ausschluss:** Erinnerungsschreiben, Postnachweise, interne Notizen oder Seiten ohne die formale Bescheidstruktur.

**AUSGABEFORMAT**

Gib ausschließlich gültiges JSON im folgenden Format zurück (keine Markdown-Codeblöcke, keine Zusatztexte):

{{
  "sections": [
    {{
      "start_page": <INT>,
      "end_page": <INT>,
      "document_type": "<Anhörung|Bescheid>",
      "confidence": <FLOAT>,
      "partial_from_previous": <BOOL>,
      "partial_into_next": <BOOL>
    }},
    ...
  ]
}}

**WEITERE HINWEISE**
- Verwende physische Seitenzahlen (erste Seite = 1).
- Identifiziere nur Dokumente mit hoher Sicherheit (confidence > 0.8 empfohlen).
- Falls ein Abschnitt über mehrere Seiten geht, gib den vollständigen Bereich an.
- Setze die Partial-Flags auf true, falls der Abschnitt offensichtlich vor oder nach dem betrachteten Bereich weiterläuft (sonst false).
- Sollte kein passender Abschnitt gefunden werden, gib "sections": [] zurück.
"""


def load_gemini_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY environment variable not set")
    return genai.Client(api_key=api_key)


def prepare_output_directory(base_dir: str) -> Path:
    output_base = Path(base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Demote previous "-current" runs
    for child in output_base.iterdir():
        if child.is_dir() and child.name.endswith("-current"):
            base_name = child.name[:-8]  # remove "-current"
            if base_name:
                target = output_base / base_name
                if target.exists():
                    shutil.rmtree(target)
                child.rename(target)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_base / f"{timestamp}-current"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def extract_pages(
    pdf_path: str, sections: DocumentSections, output_dir: Path
) -> List[str]:
    if not sections.sections:
        print("ℹ️ Keine Abschnitte zum Extrahieren gefunden")
        return []

    filename_base = Path(pdf_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files: List[str] = []

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

        for idx, section in enumerate(sections.sections, start=1):
            start_idx = section.start_page - 1
            end_idx = section.end_page - 1

            if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                print(
                    f"⚠️ Abschnitt {idx}: Ungültiger Seitenbereich "
                    f"{section.start_page}-{section.end_page}, übersprungen"
                )
                continue

            doc_type_clean = section.document_type.replace(" ", "_")
            output_filename = (
                f"{filename_base}_{doc_type_clean}_p{section.start_page}-{section.end_page}.pdf"
            )
            output_file = output_dir / output_filename

            new_pdf = pikepdf.Pdf.new()
            for page_idx in range(start_idx, end_idx + 1):
                new_pdf.pages.append(pdf_doc.pages[page_idx])

            new_pdf.save(output_file)
            extracted_files.append(str(output_file))

            continuation = []
            if section.partial_from_previous:
                continuation.append("extends backward")
            if section.partial_into_next:
                continuation.append("extends forward")

            continuation_text = f" ({', '.join(continuation)})" if continuation else ""
            print(
                f"  ✅ Abschnitt {idx} ({section.document_type}): "
                f"Seiten {section.start_page}-{section.end_page}, "
                f"Confidence {section.confidence:.2f}{continuation_text}"
            )

    return extracted_files


def parse_response_text(text: str) -> DocumentSections:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return DocumentSections.model_validate_json(cleaned)
    except Exception as exc:
        raise ValueError(f"Antwort konnte nicht geparst werden: {exc}\nRohtext:\n{cleaned}")


def identify_sections_with_gemini(
    client: genai.Client, pdf_path: str, total_pages: int, config: GeminiConfig
) -> DocumentSections:
    print(f"📁 {os.path.basename(pdf_path)}: Lade PDF zu Gemini hoch...")
    with open(pdf_path, "rb") as pdf_file:
        uploaded = client.files.upload(
            file=pdf_file,
            config={
                "mime_type": "application/pdf",
                "display_name": os.path.basename(pdf_path),
            },
        )

    prompt = PROMPT_TEMPLATE.format(total_pages=total_pages)

    response = client.models.generate_content(
        model=config.model,
        contents=[
            prompt,
            uploaded,
        ],
        config=types.GenerateContentConfig(temperature=config.temperature),
    )

    response_text = response.text or ""
    if not response_text.strip():
        raise ValueError("Gemini lieferte keinen Text zurück")

    print("🧾 Rohantwort von Gemini:")
    print(response_text.strip())

    sections = parse_response_text(response_text)

    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        if any(x is not None for x in (input_tokens, output_tokens, total_tokens)):
            print("📊 Token-Verbrauch (Gemini):")
            if input_tokens is not None:
                print(f"  ├─ Input tokens:  {input_tokens}")
            if output_tokens is not None:
                print(f"  ├─ Output tokens: {output_tokens}")
            if total_tokens is not None:
                print(f"  └─ Total tokens: {total_tokens}")

    client.files.delete(name=uploaded.name)
    print(f"🗑️ {os.path.basename(pdf_path)}: Hochgeladene Datei entfernt")

    return sections


def segment_pdf_with_gemini(
    pdf_path: str,
    output_dir: Path,
    client: Optional[genai.Client] = None,
    config: Optional[GeminiConfig] = None,
    verbose: bool = True,
) -> Tuple[DocumentSections, List[Tuple[PageRange, str]]]:
    """Identify sections and extract them into individual PDF files."""

    if client is None:
        client = load_gemini_client()
    if config is None:
        config = GeminiConfig()

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    sections = identify_sections_with_gemini(client, pdf_path, total_pages, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted: List[Tuple[PageRange, str]] = []
    filename_base = Path(pdf_path).stem

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

        for idx, section in enumerate(sections.sections, start=1):
            start_idx = section.start_page - 1
            end_idx = section.end_page - 1

            if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                if verbose:
                    print(
                        f"⚠️ Abschnitt {idx}: Ungültiger Seitenbereich "
                        f"{section.start_page}-{section.end_page}, übersprungen"
                    )
                continue

            doc_type_clean = section.document_type.replace(" ", "_")
            output_filename = (
                f"{filename_base}_{doc_type_clean}_p{section.start_page}-{section.end_page}.pdf"
            )
            output_file = output_dir / output_filename

            new_pdf = pikepdf.Pdf.new()
            for page_idx in range(start_idx, end_idx + 1):
                new_pdf.pages.append(pdf_doc.pages[page_idx])

            new_pdf.save(output_file)
            extracted.append((section, str(output_file)))

            if verbose:
                continuation = []
                if section.partial_from_previous:
                    continuation.append("extends backward")
                if section.partial_into_next:
                    continuation.append("extends forward")
                continuation_text = f" ({', '.join(continuation)})" if continuation else ""
                print(
                    f"  ✅ Abschnitt {idx} ({section.document_type}): "
                    f"Seiten {section.start_page}-{section.end_page}, "
                    f"Confidence {section.confidence:.2f}{continuation_text}"
                )

    return sections, extracted


def process_document(
    client: genai.Client,
    pdf_path: str,
    output_dir: Path,
    config: GeminiConfig,
) -> Tuple[DocumentSections, List[str]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF nicht gefunden: {pdf_path}")

    filename = os.path.basename(pdf_path)
    print("\n" + "=" * 60)
    print(f"📋 Verarbeitung: {filename}")
    print("=" * 60)

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    print(f"📄 Gesamtseiten: {total_pages}")
    start_time = time.time()

    model_output_dir = output_dir / "gemini-2.5-flash-preview-09-2025"
    sections, extracted_pairs = segment_pdf_with_gemini(
        pdf_path, model_output_dir, client=client, config=config, verbose=True
    )
    extracted_files = [path for _, path in extracted_pairs]

    elapsed = time.time() - start_time
    print(f"\n✅ Verarbeitung abgeschlossen in {elapsed:.1f} Sekunden")
    print(f"📁 {len(extracted_files)} Datei(en) gespeichert in {model_output_dir}")

    return sections, extracted_files


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extrahiere Anhörung- und Bescheid-Abschnitte aus PDF-Dokumenten "
            "mit Gemini 2.5 Flash Preview"
        )
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="Pfad(e) zu den zu analysierenden PDF-Dateien",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="extracted_documents_gemini",
        help="Basisverzeichnis für extrahierte PDFs (Standard: extracted_documents_gemini)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        client = load_gemini_client()
    except Exception as exc:
        print(f"❌ Konnte Gemini-Client nicht initialisieren: {exc}")
        sys.exit(1)

    run_dir = prepare_output_directory(args.output_dir)
    print(f"📂 Ergebnisse werden in {run_dir} abgelegt")

    config = GeminiConfig()
    all_results: dict[str, DocumentSections] = {}

    for pdf_path in args.pdf_paths:
        try:
            sections, extracted_files = process_document(
                client, pdf_path, run_dir, config
            )
            all_results[pdf_path] = sections
        except Exception as exc:
            print(f"❌ Fehler bei {pdf_path}: {exc}")

    print("\n" + "=" * 60)
    print("📊 Zusammenfassung")
    print("=" * 60)

    if not all_results:
        print("Keine Dokumente erfolgreich verarbeitet.")
        return

    for pdf_path, sections in all_results.items():
        filename = os.path.basename(pdf_path)
        print(f"\n📁 {filename}")
        if sections.sections:
            for section in sections.sections:
                continuation = []
                if section.partial_from_previous:
                    continuation.append("↤")
                if section.partial_into_next:
                    continuation.append("↦")
                continuation_text = f" {' '.join(continuation)}" if continuation else ""
                print(
                    f"  • {section.document_type} "
                    f"(Seiten {section.start_page}-{section.end_page}, "
                    f"Confidence {section.confidence:.2f}){continuation_text}"
                )
        else:
            print("  Keine Abschnitte erkannt.")

    print(f"\n📂 Aktueller Lauf gespeichert unter: {run_dir}")


if __name__ == "__main__":
    main()
