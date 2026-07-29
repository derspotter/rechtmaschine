"""Outline-based PDF segmentation helpers (Akte -> Anhörung/Bescheid) and
PDF chunking for uploads.

Der frühere Gemini-Segmentierungspfad (ganze Akten-PDFs an Google) wurde am
29.07.2026 entfernt (Datenschutzkonzept M3/M4): Segmentierung läuft
ausschließlich lokal (PDF-Outline hier, Qwen-Vision in
document_segmentation.py).
"""

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pikepdf
import pydantic

from extract_anhoerungen_bescheide import (
    collect_outline_items,
    normalize_text,
    select_items,
    select_items_by_includes,
)


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
    outline_title: Optional[str] = pydantic.Field(
        default=None,
        description="Original PDF outline/TOC title used for detection",
    )
    hearing_subtype: Optional[str] = pydantic.Field(
        default=None,
        description="Derived hearing subtype such as 'dublin', 'zulassigkeit', or 'substantive'",
    )


class DocumentSections(pydantic.BaseModel):
    """Identified sections in the legal document."""

    sections: List[PageRange] = pydantic.Field(
        default_factory=list,
        description="List of identified Anhörung and Bescheid sections",
    )


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

            doc_type_clean = _build_segment_filename_label(section.document_type, section.hearing_subtype)
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


def _infer_document_type_from_title(title: str) -> str:
    title_norm = normalize_text(title)
    if "bescheid" in title_norm:
        return "Bescheid"
    if any(
        token in title_norm
        for token in ("anhorung", "anhoerung", "niederschrift", "erstbefragung", "zweitbefragung", "befragung")
    ):
        return "Anhörung"
    return "Sonstige gespeicherte Quellen"


def _infer_hearing_subtype_from_title(title: str) -> Optional[str]:
    title_norm = normalize_text(title)
    if "dublin" in title_norm:
        return "dublin"
    if any(token in title_norm for token in ("zulass", "erstbefragung")):
        return "zulassigkeit"
    if any(
        token in title_norm
        for token in ("anhorung", "anhoerung", "niederschrift", "zweitbefragung", "befragung")
    ):
        return "substantive"
    return None


def _build_segment_filename_label(document_type: str, hearing_subtype: Optional[str]) -> str:
    label = document_type.replace(" ", "_")
    if document_type == "Anhörung" and hearing_subtype:
        subtype_label = {
            "dublin": "Dublin",
            "zulassigkeit": "Zulaessigkeit",
            "substantive": "Materiell",
        }.get(hearing_subtype, hearing_subtype.title())
        label = f"{label}_{subtype_label}"
    return label


def _is_default_outline_match(title: str) -> bool:
    """Select likely hearing/decision sections from BAMF Akten outlines."""

    title_norm = normalize_text(title)

    if re.search(r"\bbescheid\b$", title_norm) or "bescheid_ablehnung" in title_norm:
        return True

    if any(
        token in title_norm
        for token in (
            "kontrollbogen",
            "checkliste",
            "sprachauffalligkeit",
            "sprachauffaelligkeit",
            "merkblatt",
            "ladung",
            "niederschriftteil",
            "dublin-erklarung",
            "dublin_erklarung",
            "dublinet",
            "kurzubersicht",
            "verfugung_bescheidzustellung",
            "bescheidzustellung",
            "rechtsbehelfsbelehrung",
            "bescheid-ubersetzung",
            "bescheid_ubersetzung",
        )
    ):
        return False

    return any(
        token in title_norm
        for token in (
            "anhorung_",
            "anhoerung_",
            "niederschrift",
            "erstbefragung",
            "zweitbefragung",
            "befragung",
        )
    )


def segment_pdf_with_outline(
    pdf_path: str,
    output_dir: Path,
    includes: Optional[List[str]] = None,
    pattern: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[DocumentSections, List[Tuple[PageRange, str]]]:
    """Identify sections via PDF outline/TOC and extract them into individual PDF files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        items = collect_outline_items(pdf_doc)
        if not items:
            if verbose:
                print("ℹ️ Keine Outline-Einträge gefunden")
            return DocumentSections(sections=[]), []

        if includes:
            selected = select_items_by_includes(items, includes)
        elif pattern:
            selected = select_items(items, pattern)
        else:
            selected = [item for item in items if _is_default_outline_match(item["title"])]

        if not selected:
            if verbose:
                print("ℹ️ Keine passenden Outline-Einträge gefunden")
            return DocumentSections(sections=[]), []

        extracted: List[Tuple[PageRange, str]] = []
        sections: List[PageRange] = []
        filename_base = Path(pdf_path).stem
        total_pages = len(pdf_doc.pages)

        for idx, item in enumerate(selected, start=1):
            start_page = item["start"] + 1
            end_page = item["end"] + 1
            doc_type = _infer_document_type_from_title(item["title"])
            confidence = 0.95 if doc_type in ("Anhörung", "Bescheid") else 0.6
            section = PageRange(
                start_page=start_page,
                end_page=end_page,
                document_type=doc_type,
                confidence=confidence,
                partial_from_previous=False,
                partial_into_next=False,
                outline_title=item["title"],
                hearing_subtype=_infer_hearing_subtype_from_title(item["title"]) if doc_type == "Anhörung" else None,
            )
            sections.append(section)

            doc_type_clean = _build_segment_filename_label(section.document_type, section.hearing_subtype)
            output_filename = (
                f"{filename_base}_{doc_type_clean}_p{section.start_page}-{section.end_page}.pdf"
            )
            output_file = output_dir / output_filename

            new_pdf = pikepdf.Pdf.new()
            for page_idx in range(item["start"], item["end"] + 1):
                if 0 <= page_idx < total_pages:
                    new_pdf.pages.append(pdf_doc.pages[page_idx])
            new_pdf.save(output_file)

            extracted.append((section, str(output_file)))

            if verbose:
                print(
                    f"  ✅ Abschnitt {idx} ({section.document_type}): "
                    f"Seiten {section.start_page}-{section.end_page} "
                    f"(TOC: {item['title']})"
                )

        return DocumentSections(sections=sections), extracted


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


