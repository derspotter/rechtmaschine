#!/usr/bin/env python3
"""
Extract Anhörung (hearing) and Bescheid (decision) pages from legal documents (Akten).
Uses OpenAI GPT models (GPT-5-mini by default, GPT-5-nano optional) to identify relevant pages
and extracts them as separate PDFs.
"""

import os
import sys
import argparse
import pikepdf
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pydantic
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# OpenAI imports
from openai import OpenAI

# Tenacity for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# Pydantic models for structured responses
class PageRange(pydantic.BaseModel):
    """A range of pages identified in the document."""
    start_page: int = pydantic.Field(description="1-based physical page index where section starts")
    end_page: int = pydantic.Field(description="1-based physical page index where section ends")
    document_type: str = pydantic.Field(description="Type of document: 'Anhörung' or 'Bescheid'")
    confidence: float = pydantic.Field(ge=0.0, le=1.0, description="Confidence in identification")
    partial_from_previous: bool = pydantic.Field(
        default=False,
        description="True if the section appears to begin before this part of the document"
    )
    partial_into_next: bool = pydantic.Field(
        default=False,
        description="True if the section appears to continue after this part of the document"
    )


class DocumentSections(pydantic.BaseModel):
    """Identified sections in the legal document."""
    sections: List[PageRange] = pydantic.Field(
        default_factory=list,
        description="List of identified Anhörung and Bescheid sections"
    )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an OpenAI model used by this script."""
    model: str
    service_tier: Optional[str] = None
    input_cost_per_million: Optional[float] = None
    cached_input_cost_per_million: Optional[float] = None
    output_cost_per_million: Optional[float] = None
    display_name: Optional[str] = None

    @property
    def label(self) -> str:
        return self.display_name or self.model


MODEL_CONFIGS = {
    "gpt-5-mini": ModelConfig(
        model="gpt-5-mini",
        service_tier="flex",
        input_cost_per_million=0.125,
        cached_input_cost_per_million=0.0125,
        output_cost_per_million=1.00,
        display_name="GPT-5-mini"
    ),
    "gpt-5-nano": ModelConfig(
        model="gpt-5-nano",
        service_tier="flex",
        input_cost_per_million=0.025,
        cached_input_cost_per_million=0.0025,
        output_cost_per_million=0.20,
        display_name="GPT-5-nano"
    ),
}

MAX_FILE_BYTES = 32 * 1024 * 1024  # Legacy constant retained for backward compatibility
MAX_CHUNK_BYTES = 10 * 1024 * 1024  # Current per-file upload limit
CHUNK_OVERLAP_PAGES = 2  # Number of pages to overlap between chunks


@dataclass
class PDFChunk:
    """Represents a chunked portion of a PDF ready for upload."""
    index: int
    start_page: int
    end_page: int
    path: str
    size_bytes: int


def chunk_pdf_for_upload(
    pdf_path: str,
    max_chunk_bytes: int = MAX_CHUNK_BYTES,
    overlap_pages: int = CHUNK_OVERLAP_PAGES
) -> List[PDFChunk]:
    """
    Split a PDF into chunks that satisfy the OpenAI upload limits.

    Args:
        pdf_path: Source PDF path.
        max_chunk_bytes: Maximum chunk size in bytes.
        overlap_pages: Pages to overlap between successive chunks.

    Returns:
        List of PDFChunk objects written to temporary files.
    """
    chunks: List[PDFChunk] = []

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)
        if total_pages == 0:
            return []

        start_page = 1
        chunk_idx = 1

        while start_page <= total_pages:
            current_pages: List[int] = []
            end_page = start_page - 1

            for page_num in range(start_page, total_pages + 1):
                current_pages.append(page_num)

                test_pdf = pikepdf.Pdf.new()
                for idx in current_pages:
                    test_pdf.pages.append(pdf_doc.pages[idx - 1])

                buffer = BytesIO()
                test_pdf.save(buffer)
                size_bytes = buffer.tell()

                if size_bytes > max_chunk_bytes:
                    current_pages.pop()
                    if not current_pages:
                        raise ValueError(
                            f"Page {page_num} alone exceeds the per-file limit of {max_chunk_bytes / (1024 * 1024):.2f} MB"
                        )
                    end_page = current_pages[-1]
                    break

                end_page = page_num

            if not current_pages:
                break

            # Build final PDF chunk from the selected pages
            chunk_pdf = pikepdf.Pdf.new()
            for idx in current_pages:
                chunk_pdf.pages.append(pdf_doc.pages[idx - 1])

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.close()
            chunk_pdf.save(temp_file.name)
            actual_size = os.path.getsize(temp_file.name)

            chunks.append(
                PDFChunk(
                    index=chunk_idx,
                    start_page=start_page,
                    end_page=end_page,
                    path=temp_file.name,
                    size_bytes=actual_size
                )
            )

            chunk_idx += 1
            if end_page >= total_pages:
                break
            next_start = end_page + 1
            if overlap_pages > 0:
                next_start = max(end_page - overlap_pages + 1, start_page + 1)
            start_page = next_start

    return chunks


def upload_chunk(
    client: OpenAI,
    chunk: PDFChunk,
    filename: str,
    chunk_count: int
) -> str:
    """
    Upload a PDF chunk to OpenAI and return the resulting file identifier.
    """
    size_mb = chunk.size_bytes / (1024 * 1024)
    print(
        f"📤 {filename}: Uploading chunk {chunk.index}/{chunk_count} "
        f"(pages {chunk.start_page}-{chunk.end_page}, {size_mb:.2f} MB)..."
    )

    with open(chunk.path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")

    print(
        f"✅ {filename}: Chunk {chunk.index}/{chunk_count} uploaded as {uploaded.id}"
    )

    return uploaded.id


def merge_sections(sections: List[PageRange]) -> List[PageRange]:
    """
    Merge overlapping or adjacent sections of the same type.

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
        if (
            current.document_type == last.document_type
            and current.start_page <= last.end_page + 1
        ):
            last.end_page = max(last.end_page, current.end_page)
            last.confidence = max(last.confidence, current.confidence)
            last.partial_from_previous = last.partial_from_previous or current.partial_from_previous
            last.partial_into_next = last.partial_into_next or current.partial_into_next
        else:
            merged.append(current)

    return merged

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((Exception,))
)
def identify_sections_with_model(
    client: OpenAI,
    uploaded_file_id: str,
    filename: str,
    total_pages: int,
    model_config: ModelConfig,
    chunk_index: int,
    total_chunks: int,
    chunk_start_page: int,
    chunk_end_page: int,
    overlap_pages: int
) -> DocumentSections:
    """
    Use an OpenAI model to identify Anhörung and Bescheid sections in the PDF.

    Args:
        client: OpenAI client instance
        uploaded_file_id: Identifier returned by the Files API for the uploaded PDF part
        filename: Base filename (used for logging)
        total_pages: Total number of pages in the full PDF
        model_config: Model configuration describing which OpenAI model to use
        chunk_index: Position of this chunk (1-based)
        total_chunks: Total number of chunks generated for the document
        chunk_start_page: First original page number contained in this chunk
        chunk_end_page: Last original page number contained in this chunk
        overlap_pages: Number of overlapping pages shared with adjacent chunks

    Returns:
        DocumentSections with identified page ranges
    """
    # Construct the shared prompt
    prompt = f"""
Analyze this legal document (Akte) and identify all pages that contain:
1. **Anhörung** (hearing notice/hearing document)
2. **Bescheid** (official decision/ruling)

The document has {total_pages} total pages, numbered 1 to {total_pages}.

This is **part {chunk_index} of {total_chunks}**, covering original pages {chunk_start_page} to {chunk_end_page}.
Adjacent parts overlap by {overlap_pages} page(s) to reduce boundary issues.

**INSTRUCTIONS (STRICT SCREENING):**
1. Screen only for fully formed documents, not mere mentions or indexes. The section must clearly begin on the page range you return.
2. Look specifically for:
   - "Anhörung" / "Anhörungsschreiben" / "Niederschrift über die Anhörung" or equivalent formal heading
   - "Bescheid" with official formatting. Notes, memos, index lists or references do **not** qualify.
3. **IMPORTANT: Use 1-based physical page indices** (first page = 1, second page = 2, etc.)
4. For each identified section, provide:
   - start_page: The 1-based page number where the document starts
   - end_page: The 1-based page number where the document ends
   - document_type: Either "Anhörung" or "Bescheid"
   - confidence: Your confidence level (0.0 to 1.0)
    - partial_from_previous: true if the section appears to begin before page {chunk_start_page}
    - partial_into_next: true if the section appears to continue beyond page {chunk_end_page}

**STRICT MATCHING RULES:**
- **Anhörung** (protocol) must exhibit **all** of the following hallmarks:
  - Official BAMF heading on the first page (coat of arms with "Bundesamt für Migration und Flüchtlinge" on the left) plus a "Bearbeitende Stelle" box on the right and an "Az:" reference near the top.
  - Title line beginning with "Niederschrift" (e.g., "Niederschrift über die Anhörung …" or "Niederschrift über die Befragung …") followed by context such as date, location, or §-reference.
  - Introductory paragraph starting with "Es erscheint der/die Antragsteller/in …" and further boilerplate about interpreter, language, Mitwirkungspflicht, etc., often including question/answer format.
  - Closing area with signature lines or acknowledgement statements showing it is the full transcript.
  - **Reject** control sheets, invitation letters, checklists, or other forms (e.g., "Kontrollbogen – Anhörung") even if they mention "Anhörung" but do not contain the transcript narrative described above.

- **Bescheid** must display classic decision characteristics:
  - Prominent centered heading "BESCHEID" (often letter-spaced "B E S C H E I D") with the BAMF coat of arms/letterhead on the first page. Top-right usually shows "Bundesamt für Migration und Flüchtlinge", location, date, and "Gesch.-Z." reference.
  - Section containing personal data of the applicant (name, birth details, AZR number, address) followed by the phrase "ergeht folgende Entscheidung:" and a numbered decision list.
  - Legal basis citations (e.g., §§ 60, 25 AsylG), references to Rechtsbehelfsbelehrung, and closing elements such as signature blocks or official footer with BAMF contact data.
  - **Reject** internal notes, reminders, postal slips, or any page lacking the formal decision framing described above even if it contains the word "Bescheid".

**OUTPUT FORMAT:**
Return a JSON structure with all identified sections. If no Anhörung or Bescheid is found, return an empty sections list.

**IMPORTANT:**
- Be conservative - only identify pages you're highly confident about (confidence > 0.8 recommended)
- If a document spans multiple pages, include all of them in the range
- The document might contain multiple Anhörungen or Bescheide - identify all of them
- If you believe a section begins before page {chunk_start_page} or continues after page {chunk_end_page}, set the corresponding partial flag(s) to true.
"""

    print(
        f"🔍 {filename}: Analyzing part {chunk_index}/{total_chunks} "
        f"(pages {chunk_start_page}-{chunk_end_page}) with {model_config.label}..."
    )

    parse_kwargs = dict(
        model=model_config.model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded_file_id},
                    {"type": "input_text", "text": prompt}
                ]
            }
        ],
        text_format=DocumentSections
    )
    if model_config.service_tier:
        parse_kwargs["service_tier"] = model_config.service_tier

    api_start = time.time()
    response = client.responses.parse(**parse_kwargs)
    api_duration = time.time() - api_start

    if hasattr(response, 'output_parsed') and response.output_parsed:
        sections = response.output_parsed
    elif hasattr(response, 'output_text') and response.output_text:
        sections = DocumentSections.model_validate_json(response.output_text)
    else:
        raise ValueError("No structured output found in response")

    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        total_tokens = getattr(usage, 'total_tokens', 0)
        cached_input_tokens = 0
        for attr in ("cached_input_tokens", "input_tokens_cached"):
            value = getattr(usage, attr, None)
            if value is not None:
                cached_input_tokens = value
                break
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if not cached_input_tokens and prompt_details:
            if isinstance(prompt_details, dict):
                cached_input_tokens = prompt_details.get("cached_tokens", 0)
            else:
                cached_input_tokens = getattr(prompt_details, "cached_tokens", 0)

        print(
            f"📊 {filename}: Part {chunk_index}/{total_chunks} token usage ({model_config.label}):"
        )
        print(f"  ├─ Input tokens: {input_tokens:,}")
        print(f"  ├─ Output tokens: {output_tokens:,}")
        print(f"  ├─ Total tokens: {total_tokens:,}")
        if cached_input_tokens:
            print(f"  ├─ Cached input tokens: {cached_input_tokens:,}")

        if model_config.input_cost_per_million is not None and model_config.output_cost_per_million is not None:
            billable_input_tokens = max(input_tokens - cached_input_tokens, 0)
            input_cost = (billable_input_tokens / 1_000_000) * model_config.input_cost_per_million
            cached_cost = 0.0
            if cached_input_tokens and model_config.cached_input_cost_per_million is not None:
                cached_cost = (cached_input_tokens / 1_000_000) * model_config.cached_input_cost_per_million
            output_cost = (output_tokens / 1_000_000) * model_config.output_cost_per_million
            total_cost = input_cost + output_cost + cached_cost
            print(f"  └─ Cost: ${total_cost:.4f}")
        else:
            print("  └─ Cost: Pricing data unavailable for this model")

    print(
        f"⏱️ {filename}: Part {chunk_index}/{total_chunks} "
        f"{model_config.label} API call took {api_duration:.2f} seconds"
    )

    return sections


def extract_pages(
    pdf_path: str,
    sections: DocumentSections,
    output_dir: str,
    model_label: Optional[str] = None
) -> List[str]:
    """
    Extract identified sections to separate PDF files.

    Args:
        pdf_path: Path to source PDF
        sections: Identified sections to extract
        output_dir: Directory to save extracted PDFs
        model_label: Optional human-readable model identifier for logging

    Returns:
        List of paths to extracted PDF files
    """
    if not sections.sections:
        print("ℹ️ No sections to extract")
        return []

    filename_base = Path(pdf_path).stem
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extracted_files = []

    label_suffix = f" [{model_label}]" if model_label else ""
    print(f"\n📄 Extracting sections from {os.path.basename(pdf_path)}{label_suffix}...")

    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

        for i, section in enumerate(sections.sections, 1):
            # Convert 1-based to 0-based indices
            start_idx = section.start_page - 1
            end_idx = section.end_page - 1

            # Validate page ranges
            if start_idx < 0 or end_idx >= total_pages or start_idx > end_idx:
                print(f"⚠️ Section {i}: Invalid page range {section.start_page}-{section.end_page}, skipping")
                continue

            # Create output filename
            doc_type_clean = section.document_type.replace(" ", "_")
            output_filename = f"{filename_base}_{doc_type_clean}_p{section.start_page}-{section.end_page}.pdf"
            output_file = output_path / output_filename

            # Extract pages
            try:
                new_pdf = pikepdf.Pdf.new()
                for page_idx in range(start_idx, end_idx + 1):
                    new_pdf.pages.append(pdf_doc.pages[page_idx])

                new_pdf.save(output_file)
                extracted_files.append(str(output_file))

                print(f"  ✅ Section {i} ({section.document_type}):")
                print(f"     ├─ Pages: {section.start_page}-{section.end_page}")
                print(f"     ├─ Confidence: {section.confidence:.2f}")
                if section.partial_from_previous or section.partial_into_next:
                    continuation_flags = []
                    if section.partial_from_previous:
                        continuation_flags.append("extends backward")
                    if section.partial_into_next:
                        continuation_flags.append("extends forward")
                    continuation_text = ", ".join(continuation_flags)
                    print(f"     ├─ Continuation: {continuation_text}")
                print(f"     └─ Saved to: {output_filename}")

            except Exception as e:
                print(f"  ❌ Section {i}: Error extracting pages: {e}")

    return extracted_files


def process_akte(
    pdf_path: str,
    total_pages: int,
    output_dir: str,
    model_config: ModelConfig,
    combined_sections: List[PageRange]
) -> Tuple[DocumentSections, List[str]]:
    """
    Finalize identified sections for a model and extract pages.

    Args:
        pdf_path: Path to the PDF file
        total_pages: Total page count for the PDF (pre-computed)
        output_dir: Directory to save extracted PDFs
        model_config: Model configuration to use
        combined_sections: Collected PageRange entries across all chunks

    Returns:
        Tuple of (identified sections, list of extracted file paths)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    merged_sections = merge_sections(combined_sections)
    sections = DocumentSections(sections=merged_sections)

    extracted_files = extract_pages(pdf_path, sections, output_dir, model_config.label)

    return sections, extracted_files


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract Anhörung and Bescheid pages from legal documents using OpenAI GPT models"
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        help="Path(s) to the PDF file(s) (Akten) to process"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="extracted_documents",
        help="Output directory for extracted PDFs (default: extracted_documents)"
    )
    parser.add_argument(
        "-m", "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=["gpt-5-mini"],
        help="Model(s) to use for analysis (default: gpt-5-mini)"
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)

    try:
        # Initialize OpenAI client once
        client = OpenAI(api_key=api_key, timeout=600.0)

        # Prepare results holder
        all_results = {}

        output_base = Path(args.output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

        # Demote any prior "-current" run by removing the suffix.
        for child in output_base.iterdir():
            if child.is_dir() and child.name.endswith("-current"):
                base_name = child.name[:-8]  # strip "-current"
                if not base_name:
                    continue
                target = output_base / base_name
                if target.exists():
                    shutil.rmtree(target)
                child.rename(target)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"{timestamp}-current"
        run_output_base = output_base / run_dir_name
        run_output_base.mkdir(parents=True, exist_ok=False)

        print(f"📂 Writing extracted PDFs to: {run_output_base}")

        for pdf_path in args.pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            filename = os.path.basename(pdf_path)

            with pikepdf.Pdf.open(pdf_path) as pdf_doc:
                total_pages = len(pdf_doc.pages)

            try:
                chunks = chunk_pdf_for_upload(pdf_path, MAX_CHUNK_BYTES, CHUNK_OVERLAP_PAGES)
            except ValueError as chunk_err:
                print(f"\n⚠️ Skipping {filename}: {chunk_err}")
                continue

            if not chunks:
                print(f"\n⚠️ Skipping {filename}: no pages available for processing.")
                continue

            pdf_start_time = time.time()

            print(f"\n{'='*60}")
            print(f"📋 Processing: {filename}")
            print(f"{'='*60}")
            print(f"📄 Total pages: {total_pages}")
            print(f"🧩 Prepared {len(chunks)} chunk(s) (≤ {MAX_CHUNK_BYTES / (1024*1024):.0f} MB each, overlap {CHUNK_OVERLAP_PAGES} page(s))")

            combined_sections: Dict[str, List[PageRange]] = {
                model_key: [] for model_key in args.models
            }
            chunk_count = len(chunks)

            with ThreadPoolExecutor(
                max_workers=min(max(len(chunks), 1), 4)
            ) as upload_executor:
                upload_futures = {
                    chunk.index: upload_executor.submit(
                        upload_chunk, client, chunk, filename, chunk_count
                    )
                    for chunk in chunks
                }

                for chunk in chunks:
                    file_id: Optional[str] = None
                    try:
                        file_id = upload_futures[chunk.index].result()
                    except Exception as upload_err:
                        print(
                            f"❌ {filename}: Upload failed for chunk {chunk.index}: {upload_err}"
                        )
                        try:
                            os.remove(chunk.path)
                        except OSError:
                            pass
                        continue

                    for model_key in args.models:
                        model_config = MODEL_CONFIGS[model_key]
                        chunk_sections = identify_sections_with_model(
                            client,
                            file_id,
                            filename,
                            total_pages,
                            model_config,
                            chunk_index=chunk.index,
                            total_chunks=chunk_count,
                            chunk_start_page=chunk.start_page,
                            chunk_end_page=chunk.end_page,
                            overlap_pages=CHUNK_OVERLAP_PAGES
                        )

                        print(
                            f"✅ {filename}: {model_config.label} part {chunk.index}/{chunk_count} "
                            f"found {len(chunk_sections.sections)} section(s)"
                        )

                        if model_config.model == "gpt-5-mini":
                            print(
                                f"🧾 {filename}: GPT-5-mini raw response (part {chunk.index}/{chunk_count}):\n"
                                f"{chunk_sections.model_dump_json(indent=2)}"
                            )

                        combined_sections[model_key].extend(chunk_sections.sections)

                    try:
                        client.files.delete(file_id)
                        print(
                            f"🗑️ {filename}: Deleted uploaded chunk {chunk.index} ({file_id})"
                        )
                    except Exception as delete_err:
                        print(
                            f"⚠️ {filename}: Warning - could not delete chunk {chunk.index}: {delete_err}"
                        )

                    try:
                        os.remove(chunk.path)
                    except OSError:
                        pass

            pdf_results = {}
            for model_key in args.models:
                model_config = MODEL_CONFIGS[model_key]
                model_output_dir = str(run_output_base / model_key)

                print(f"\n🤖 {model_config.label}: finalizing merged sections...")
                sections, extracted_files = process_akte(
                    pdf_path,
                    total_pages,
                    model_output_dir,
                    model_config,
                    combined_sections[model_key]
                )

                print(
                    f"✅ {model_config.label}: {len(sections.sections)} section(s) total, "
                    f"{len(extracted_files)} file(s) extracted"
                )

                pdf_results[model_key] = {
                    "sections": sections,
                    "extracted_files": extracted_files
                }

            total_elapsed = time.time() - pdf_start_time
            print(f"\n⏱️ Total time for {filename}: {total_elapsed:.1f} seconds")

            all_results[pdf_path] = pdf_results

        # Consolidated summary for quick comparison
        print("\n" + "=" * 60)
        print("📊 Comparison Summary")
        print("=" * 60)
        for pdf_path, models_data in all_results.items():
            print(f"\n📁 {pdf_path}")
            for model_key, result in models_data.items():
                sections = result["sections"].sections
                extracted_files = result["extracted_files"]
                model_label = MODEL_CONFIGS[model_key].label
                print(f"  🤖 {model_label}: {len(sections)} section(s) identified")
                if sections:
                    for section in sections:
                        continuation_flags = []
                        if section.partial_from_previous:
                            continuation_flags.append("↤")
                        if section.partial_into_next:
                            continuation_flags.append("↦")
                        continuation = f" {' '.join(continuation_flags)}" if continuation_flags else ""
                        print(
                            f"     • {section.document_type} "
                            f"(pages {section.start_page}-{section.end_page}, "
                            f"confidence: {section.confidence:.2f}){continuation}"
                        )
                if extracted_files:
                    print("     Extracted files:")
                    for filepath in extracted_files:
                        print(f"       - {filepath}")
                else:
                    print("     No sections extracted")


        print(f"\n📂 Completed run stored in: {run_output_base}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
