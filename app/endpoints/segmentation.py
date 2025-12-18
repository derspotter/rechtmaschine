#!/usr/bin/env python3
"""
Extract Anhörung (hearing) and Bescheid (decision) sections from PDF documents
using Google's Gemini 2.5 Flash Preview model.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import pikepdf
import pydantic
from google import genai
from google.genai import types
import shutil


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
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.0


# Gemini API limits
MAX_CHUNK_BYTES = 50 * 1024 * 1024  # 50MB limit for Gemini
CHUNK_OVERLAP_PAGES = 20  # Overlap pages to catch documents at split points


@dataclass
class PDFChunk:
    """Represents a chunked portion of a PDF ready for upload."""
    index: int
    start_page: int
    end_page: int
    path: str
    size_bytes: int


PROMPT_TEMPLATE = """Analysiere dieses deutsche BAMF-Verwaltungsdokument. {chunk_info}Identifiziere alle vollständigen Abschnitte, die eindeutig zu folgenden Kategorien gehören:

1. **Anhörung** (Anhörungsprotokoll / Niederschrift)
2. **Bescheid** (BAMF-Bescheid / Verwaltungsentscheidung)

**STRIKTE ANFORDERUNGEN**

- **Anhörung (Niederschrift):**
  - Erkennbarer BAMF-Briefkopf mit Bundesadler und „Bundesamt für Migration und Flüchtlinge“.
  - Rechts oben eine Box „Bearbeitende Stelle“ (Referat, Anschriften, Telefon/Fax) sowie ein „Az:“ (Aktenzeichen) in der Nähe der Kopfzeile.
  - Titel beginnt mit „Niederschrift …“ (z.\u202fB. „Niederschrift über die Anhörung …“ oder „Niederschrift über die Befragung …“).
  - Der Text beginnt mit Formulierungen wie „Es erscheint der/die Antragsteller/in …“, beschreibt Dolmetscher, Sprache, Belehrungen und besitzt einen Frage/Antwort- oder Fließtext-Dialog zur Anhörung.
  - Am Ende befinden sich Unterschrifts- bzw. Bestätigungsfelder.
  - **Ausschluss:** Einladungen, Kontrollbögen, Checklisten, bloße Erwähnungen oder Dokumente ohne vollständigen Anhörungs-Textkörper.

- **Bescheid (BAMF-Verfügung):**
  - Auffällige, zentrierte Überschrift „BESCHEID“ (häufig als „B E S C H E I D“) mit BAMF-Briefkopf und Bundestitel, sowie Angaben rechts oben (Ort, Datum, „Gesch.-Z.“).
  - Abschnitt mit Personalien des Antragstellers (Name, Geburtsdatum, Adresse, AZR-Nummer) gefolgt von „ergeht folgende Entscheidung:“ und nummerierter Entscheidungsliste.
  - Verweise auf Rechtsgrundlagen (§§ AsylG/AufenthG), Rechtsbehelfsbelehrung und amtliche Fuß- oder Kopfzeilen.
  - **Ausschluss:** Erinnerungsschreiben, Postnachweise, interne Notizen oder Seiten ohne die formale Bescheidstruktur.

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


def chunk_pdf_for_upload(
    pdf_path: str,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
    overlap_pages: int = CHUNK_OVERLAP_PAGES
) -> List[PDFChunk]:
    """
    Split a PDF into chunks if it exceeds Gemini's 50MB limit.
    Recursively splits in half with overlap until all chunks fit.

    Args:
        pdf_path: Source PDF path.
        max_chunk_bytes: Maximum chunk size in bytes (default 50MB).
        overlap_pages: Pages to overlap between chunks (default 20).

    Returns:
        List of PDFChunk objects.
    """
    file_size = os.path.getsize(pdf_path)
    file_size_mb = file_size / (1024 * 1024)

    # If file is under limit, return single chunk with original file
    if file_size <= max_chunk_bytes:
        with pikepdf.Pdf.open(pdf_path) as pdf_doc:
            total_pages = len(pdf_doc.pages)
        return [PDFChunk(
            index=1,
            start_page=1,
            end_page=total_pages,
            path=pdf_path,
            size_bytes=file_size
        )]

    print(f"📏 PDF ist {file_size_mb:.1f}MB (Limit: {max_chunk_bytes / (1024 * 1024):.0f}MB)")
    print(f"✂️ Teile mit {overlap_pages}-Seiten Überlappung...")

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)
        if total_pages == 0:
            return []

        # Recursive helper to split a page range
        def split_range(start_page: int, end_page: int, chunk_index: int) -> List[PDFChunk]:
            """Recursively split a page range until chunks are small enough."""
            # Build PDF for this range
            chunk_pdf = pikepdf.Pdf.new()
            for page_idx in range(start_page - 1, end_page):
                chunk_pdf.pages.append(pdf_doc.pages[page_idx])

            temp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_chunk{chunk_index}.pdf")
            temp.close()
            chunk_pdf.save(temp.name)
            size = os.path.getsize(temp.name)
            size_mb = size / (1024 * 1024)

            # If small enough, return this chunk
            if size <= max_chunk_bytes:
                print(f"  📄 Chunk {chunk_index}: Seiten {start_page}-{end_page} ({size_mb:.1f}MB)")
                return [PDFChunk(
                    index=chunk_index,
                    start_page=start_page,
                    end_page=end_page,
                    path=temp.name,
                    size_bytes=size
                )]

            # Too big - split in half and recurse
            print(f"  ⚠️ Chunk {chunk_index} ({size_mb:.1f}MB) zu groß, teile weiter...")
            os.remove(temp.name)  # Don't need this oversized chunk

            import math
            mid_page = start_page + (end_page - start_page) // 2

            # First half: start to mid
            chunks_left = split_range(start_page, mid_page, chunk_index * 2 - 1)

            # Second half: (mid - overlap) to end
            overlap_start = max(start_page, mid_page - overlap_pages + 1)
            chunks_right = split_range(overlap_start, end_page, chunk_index * 2)

            return chunks_left + chunks_right

        # Start recursive splitting
        all_chunks = split_range(1, total_pages, 1)

        # Re-index chunks sequentially
        for i, chunk in enumerate(all_chunks, start=1):
            all_chunks[i-1] = PDFChunk(
                index=i,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                path=chunk.path,
                size_bytes=chunk.size_bytes
            )

        return all_chunks


def merge_sections(sections: List[PageRange]) -> List[PageRange]:
    """
    Merge overlapping or adjacent sections of the same type.
    Needed when chunking with overlap causes duplicate detections.

    Args:
        sections: List of PageRange entries.

    Returns:
        Consolidated list of PageRange entries.
    """
    if not sections:
        return []

    sections_sorted = sorted(sections, key=lambda s: (s.start_page, s.end_page))
    merged: List[PageRange] = [sections_sorted[0]]

    for current in sections_sorted[1:]:
        last = merged[-1]
        # Merge if same type and overlapping/adjacent
        if (
            current.document_type == last.document_type
            and current.start_page <= last.end_page + 1
        ):
            # Extend the last section
            merged[-1] = PageRange(
                start_page=last.start_page,
                end_page=max(last.end_page, current.end_page),
                document_type=last.document_type,
                confidence=max(last.confidence, current.confidence),
                partial_from_previous=last.partial_from_previous,
                partial_into_next=current.partial_into_next or last.partial_into_next,
            )
        else:
            merged.append(current)

    return merged


async def process_single_chunk_async(
    client: genai.Client,
    chunk: PDFChunk,
    total_chunks: int,
    config: GeminiConfig,
) -> DocumentSections:
    """Process a single PDF chunk with Gemini asynchronously."""
    chunk_name = f"Chunk {chunk.index}/{total_chunks}" if total_chunks > 1 else os.path.basename(chunk.path)
    chunk_pages = chunk.end_page - chunk.start_page + 1

    print(f"📁 {chunk_name}: Lade zu Gemini hoch (Seiten {chunk.start_page}-{chunk.end_page})...")
    print(f"   DEBUG: Chunk hat {chunk_pages} Seiten, Original-Seiten {chunk.start_page}-{chunk.end_page}")

    with open(chunk.path, "rb") as pdf_file:
        uploaded = await client.aio.files.upload(
            file=pdf_file,
            config={
                "mime_type": "application/pdf",
                "display_name": chunk_name,
            },
        )

    # Build chunk info string for prompt
    if total_chunks > 1:
        chunk_info = f"Dieses PDF ist ein Teil (Seiten {chunk.start_page}-{chunk.end_page}) eines größeren Dokuments. Das PDF selbst hat {chunk_pages} Seiten. WICHTIG: Gib Seitenzahlen relativ zum Anfang dieses PDFs an (Seite 1-{chunk_pages}), NICHT die Original-Seitenzahlen. "
    else:
        chunk_info = f"Dieses PDF hat insgesamt {chunk_pages} physisch nummerierte Seiten (1-basiert). "

    prompt = PROMPT_TEMPLATE.format(chunk_info=chunk_info)

    try:
        response = await client.aio.models.generate_content(
            model=config.model,
            contents=[
                prompt,
                uploaded,
            ],
            config=types.GenerateContentConfig(
                temperature=config.temperature,
                response_mime_type="application/json",
                response_schema=DocumentSections,
            ),
        )
    finally:
        await client.aio.files.delete(name=uploaded.name)

    response_text = response.text or ""
    parsed_sections = getattr(response, "parsed", None)

    if parsed_sections is not None:
        if total_chunks > 1:
            print(f"🧾 {chunk_name}: Gemini Antwort:")
        else:
            print("🧾 Rohantwort von Gemini (Schema):")
        print(json.dumps(parsed_sections.model_dump(), indent=2, ensure_ascii=False))
        sections = parsed_sections
    else:
        if not response_text.strip():
            raise ValueError("Gemini lieferte keinen Text zurück")

        print("🧾 Rohantwort von Gemini (Text):")
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

    if total_chunks > 1:
        print(f"✅ {chunk_name} verarbeitet")
    else:
        print(f"🗑️ {os.path.basename(chunk.path)}: Hochgeladene Datei entfernt")

    return sections


def identify_sections_with_gemini(
    client: genai.Client, pdf_path: str, total_pages: int, config: GeminiConfig
) -> DocumentSections:
    """
    Identify sections in a PDF using Gemini.
    Automatically splits large files (>50MB) into chunks and processes them concurrently.
    """
    # Split PDF into chunks if needed
    chunks = chunk_pdf_for_upload(pdf_path)
    temp_files_to_cleanup = []

    try:
        # Track temp files for cleanup
        for chunk in chunks:
            if chunk.path != pdf_path:
                temp_files_to_cleanup.append(chunk.path)

        # Process chunks concurrently with asyncio
        if len(chunks) > 1:
            print(f"⚡ Verarbeite {len(chunks)} Chunks parallel mit asyncio...")

        all_sections: List[PageRange] = asyncio.run(_process_chunks_async(client, chunks, config))

        # Merge overlapping sections (from chunk overlap)
        if len(chunks) > 1:
            all_sections = merge_sections(all_sections)
            print(f"\n✅ Alle {len(chunks)} Chunks verarbeitet, {len(all_sections)} Abschnitte gefunden (nach Merge)")

        return DocumentSections(sections=all_sections)

    finally:
        # Clean up temporary chunk files
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ Konnte temporäre Datei nicht löschen: {temp_file} ({e})")


async def _process_chunks_async(
    client: genai.Client, chunks: List[PDFChunk], config: GeminiConfig
) -> List[PageRange]:
    """Process all chunks concurrently with asyncio.gather."""
    try:
        # Create tasks for all chunks
        tasks = [
            process_single_chunk_async(client, chunk, len(chunks), config)
            for chunk in chunks
        ]

        # Run all tasks concurrently
        chunk_results = await asyncio.gather(*tasks)

        # Adjust page numbers and collect all sections
        all_sections: List[PageRange] = []
        for chunk, chunk_sections in zip(chunks, chunk_results):
            page_offset = chunk.start_page - 1

            for section in chunk_sections.sections:
                adjusted_section = PageRange(
                    start_page=section.start_page + page_offset,
                    end_page=section.end_page + page_offset,
                    document_type=section.document_type,
                    confidence=section.confidence,
                    partial_from_previous=section.partial_from_previous,
                    partial_into_next=section.partial_into_next,
                )
                all_sections.append(adjusted_section)

        return all_sections
    finally:
        # Close the async client session properly
        await client.aio.aclose()


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

    model_output_dir = output_dir / "gemini-3-flash-preview"
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
