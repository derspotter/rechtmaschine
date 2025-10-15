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
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import pydantic

# OpenAI imports
from openai import OpenAI

# Tenacity for retry logic
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# Load environment variables
load_dotenv()

# Pydantic models for structured responses
class PageRange(pydantic.BaseModel):
    """A range of pages identified in the document."""
    start_page: int = pydantic.Field(description="1-based physical page index where section starts")
    end_page: int = pydantic.Field(description="1-based physical page index where section ends")
    document_type: str = pydantic.Field(description="Type of document: 'Anhörung' or 'Bescheid'")
    confidence: float = pydantic.Field(ge=0.0, le=1.0, description="Confidence in identification")
    evidence: str = pydantic.Field(description="Text evidence supporting the identification")


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


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((Exception,))
)
def identify_sections_with_model(
    client: OpenAI,
    pdf_path: str,
    total_pages: int,
    model_config: ModelConfig
) -> DocumentSections:
    """
    Use an OpenAI model to identify Anhörung and Bescheid sections in the PDF.

    Args:
        client: OpenAI client instance
        pdf_path: Path to the PDF file
        total_pages: Total number of pages in the PDF
        model_config: Model configuration describing which OpenAI model to use

    Returns:
        DocumentSections with identified page ranges
    """
    filename = os.path.basename(pdf_path)
    print(f"📁 {filename}: Uploading to OpenAI for analysis with {model_config.label}...")

    # Upload the full PDF
    with open(pdf_path, "rb") as f:
        uploaded_file = client.files.create(file=f, purpose="user_data")

    print(f"✅ {filename}: Uploaded as {uploaded_file.id}")

    try:
        # Construct the prompt for GPT-5-mini
        prompt = f"""
Analyze this legal document (Akte) and identify all pages that contain:
1. **Anhörung** (hearing notice/hearing document)
2. **Bescheid** (official decision/ruling)

The document has {total_pages} total pages, numbered 1 to {total_pages}.

**INSTRUCTIONS:**
1. Look for documents titled or containing:
   - "Anhörung" / "Anhörungsschreiben" / "Hearing"
   - "Bescheid" / "Verwaltungsakt" / "Decision" / "Ruling"
2. **IMPORTANT: Use 1-based physical page indices** (first page = 1, second page = 2, etc.)
3. For each identified section, provide:
   - start_page: The 1-based page number where the document starts
   - end_page: The 1-based page number where the document ends
   - document_type: Either "Anhörung" or "Bescheid"
   - confidence: Your confidence level (0.0 to 1.0)
   - evidence: Brief text excerpt or description that confirms the identification

**WHAT TO LOOK FOR:**
- **Anhörung**: Usually contains phrases like:
  - "Anhörung gemäß..."
  - "Sie haben die Möglichkeit, sich zu äußern"
  - "Anhörungsschreiben"
  - References to legal provisions for hearing rights (e.g., "§ 28 VwVfG")
  - Often shorter, inviting statement or response

- **Bescheid**: Has a very specific formal structure:
  - **CRITICAL: Large, centered heading "BESCHEID" or "B E S C H E I D"**
  - Official letterhead (e.g., "Bundesamt für Migration und Flüchtlinge")
  - File reference numbers (Geschäfts-Z., AZR-Nummer)
  - Addressed to specific persons with birth details (geb. am...)
  - Formal decision phrase: "ergeht folgende Entscheidung" (the following decision is issued)
  - Numbered decision points (1., 2., 3., etc.) with specific outcomes:
    - "offensichtlich unbegründet abgelehnt" (manifestly unfounded rejected)
    - "wird zuerkannt/abgelehnt" (is granted/rejected)
    - References to specific laws (§ 60 Abs. 5, Aufenthaltsgesetz, etc.)
  - Legal remedies section ("Rechtsbehelfsbelehrung")
  - Information about deportation possibilities
  - Formal signatures and official stamps
  - Multiple pages typical

**OUTPUT FORMAT:**
Return a JSON structure with all identified sections. If no Anhörung or Bescheid is found, return an empty sections list.

**IMPORTANT:**
- Be conservative - only identify pages you're confident about (confidence > 0.7)
- If a document spans multiple pages, include all of them in the range
- The document might contain multiple Anhörungen or Bescheide - identify all of them
"""

        # Make API request using Responses API
        print(f"🔍 {filename}: Analyzing document with {model_config.label}...")

        parse_kwargs = dict(
            model=model_config.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": uploaded_file.id},
                        {"type": "input_text", "text": prompt}
                    ]
                }
            ],
            text_format=DocumentSections
        )
        if model_config.service_tier:
            parse_kwargs["service_tier"] = model_config.service_tier

        response = client.responses.parse(**parse_kwargs)

        # Parse the structured response
        if hasattr(response, 'output_parsed') and response.output_parsed:
            sections = response.output_parsed
        elif hasattr(response, 'output_text') and response.output_text:
            sections = DocumentSections.model_validate_json(response.output_text)
        else:
            raise ValueError("No structured output found in response")

        # Display token usage if available
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

            print(f"📊 {filename}: Token usage ({model_config.label}):")
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

        print(f"✅ {filename}: Found {len(sections.sections)} section(s) with {model_config.label}")

        return sections

    finally:
        # Cleanup uploaded file
        try:
            client.files.delete(uploaded_file.id)
            print(f"🗑️ {filename}: Cleaned up uploaded file ({model_config.label})")
        except Exception as e:
            print(f"⚠️ {filename}: Warning - could not delete uploaded file: {e}")


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
                print(f"     ├─ Evidence: {section.evidence[:80]}...")
                print(f"     └─ Saved to: {output_filename}")

            except Exception as e:
                print(f"  ❌ Section {i}: Error extracting pages: {e}")

    return extracted_files


def process_akte(
    pdf_path: str,
    output_dir: str,
    client: OpenAI,
    model_config: ModelConfig
) -> Tuple[DocumentSections, List[str]]:
    """
    Process a legal document (Akte) to identify and extract Anhörung and Bescheid sections.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted PDFs
        client: OpenAI client
        model_config: Model configuration to use

    Returns:
        Tuple of (identified sections, list of extracted file paths)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    print(f"\n{'='*60}")
    print(f"📋 Processing: {filename}")
    print(f"{'='*60}")

    start_time = time.time()

    # Get total page count
    with pikepdf.Pdf.open(pdf_path) as pdf_doc:
        total_pages = len(pdf_doc.pages)

    print(f"📄 Total pages: {total_pages}")
    print(f"🤖 Model: {model_config.label}")

    # Identify sections using configured model
    sections = identify_sections_with_model(client, pdf_path, total_pages, model_config)

    # Extract identified sections
    extracted_files = extract_pages(pdf_path, sections, output_dir, model_config.label)

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"✅ Processing complete!")
    print(f"⏱️ Time: {elapsed:.1f} seconds")
    print(f"📁 Extracted {len(extracted_files)} file(s) to: {output_dir}")
    print(f"{'='*60}\n")

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

        for pdf_path in args.pdf_paths:
            pdf_results = {}
            for model_key in args.models:
                model_config = MODEL_CONFIGS[model_key]
                model_output_dir = str(Path(args.output_dir) / model_key)

                sections, extracted_files = process_akte(
                    pdf_path,
                    model_output_dir,
                    client,
                    model_config
                )

                pdf_results[model_key] = {
                    "sections": sections,
                    "extracted_files": extracted_files
                }

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
                        print(
                            f"     • {section.document_type} "
                            f"(pages {section.start_page}-{section.end_page}, "
                            f"confidence: {section.confidence:.2f})"
                        )
                if extracted_files:
                    print("     Extracted files:")
                    for filepath in extracted_files:
                        print(f"       - {filepath}")
                else:
                    print("     No sections extracted")

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
