import json
import re
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pikepdf
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
import anthropic
import traceback
import os
from openai import OpenAI
from google import genai
from google.genai import types

from shared import (
    DocumentCategory,
    GenerationRequest,
    GenerationResponse,
    get_document_for_upload,
    GenerationMetadata,
    JLawyerSendRequest,
    JLawyerResponse,
    JLawyerTemplatesResponse,
    SavedSource,
    broadcast_documents_snapshot,
    get_anthropic_client,
    get_openai_client,
    limiter,
    load_document_text,
    store_document_text,
    get_gemini_client,
)
from database import get_db
from models import Document, ResearchSource

router = APIRouter()


def _document_to_context_dict(doc) -> Dict[str, Optional[str]]:
    """Convert a Document ORM instance into a context dictionary used for prompting."""
    from pathlib import Path
    stored_path = Path(doc.file_path) if doc.file_path else None
    if stored_path and not stored_path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Dokument {doc.filename} wurde nicht auf dem Server gefunden")

    return {
        "id": str(doc.id),
        "filename": doc.filename,
        "category": doc.category,
        "file_path": str(stored_path) if stored_path else None,
        "confidence": doc.confidence,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "explanation": doc.explanation,
    }


def _validate_category(doc, expected_category: str) -> None:
    """Ensure a document matches the expected category."""
    if doc.category != expected_category:
        raise HTTPException(
            status_code=400,
            detail=f"Dokument {doc.filename} gehört zur Kategorie '{doc.category}', erwartet war '{expected_category}'",
        )


def _collect_selected_documents(selection, db: Session) -> Dict[str, List[Dict[str, Optional[str]]]]:
    """Validate and collect document metadata for Klagebegründung generation."""
    bescheid_selection = selection.bescheid

    total_selected = len(selection.anhoerung) + len(selection.rechtsprechung) + len(selection.saved_sources)
    total_selected += 1 if bescheid_selection.primary else 0
    total_selected += len(bescheid_selection.others)

    if total_selected == 0:
        raise HTTPException(status_code=400, detail="Bitte wählen Sie mindestens ein Dokument aus")

    collected: Dict[str, List[Dict[str, Optional[str]]]] = {
        "anhoerung": [],
        "bescheid": [],
        "rechtsprechung": [],
        "saved_sources": [],
    }

    # Load Anhörung documents
    if selection.anhoerung:
        query = (
            db.query(Document)
            .filter(Document.filename.in_(selection.anhoerung))
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.anhoerung if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Anhörung-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.ANHOERUNG.value)
            collected["anhoerung"].append(_document_to_context_dict(doc))

    # Load Bescheid documents (primary + others)
    if not bescheid_selection.primary:
        raise HTTPException(status_code=400, detail="Bitte markieren Sie einen Bescheid als Hauptbescheid (Anlage K2)")

    bescheid_filenames = [bescheid_selection.primary] + (bescheid_selection.others or [])
    bescheid_query = (
        db.query(Document)
        .filter(Document.filename.in_(bescheid_filenames))
        .all()
    )
    bescheid_map = {doc.filename: doc for doc in bescheid_query}

    missing_bescheide = [fn for fn in bescheid_filenames if fn not in bescheid_map]
    if missing_bescheide:
        raise HTTPException(status_code=404, detail=f"Bescheid-Dokumente nicht gefunden: {', '.join(missing_bescheide)}")

    primary_doc = bescheid_map[bescheid_selection.primary]
    _validate_category(primary_doc, DocumentCategory.BESCHEID.value)
    collected["bescheid"].append({**_document_to_context_dict(primary_doc), "role": "primary"})

    for other_name in bescheid_selection.others or []:
        doc = bescheid_map[other_name]
        _validate_category(doc, DocumentCategory.BESCHEID.value)
        collected["bescheid"].append({**_document_to_context_dict(doc), "role": "secondary"})

    # Load Rechtsprechung documents
    if selection.rechtsprechung:
        query = (
            db.query(Document)
            .filter(Document.filename.in_(selection.rechtsprechung))
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.rechtsprechung if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Rechtsprechung-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.RECHTSPRECHUNG.value)
            collected["rechtsprechung"].append(_document_to_context_dict(doc))

    # Load saved sources
    if selection.saved_sources:
        collected_sources = []
        for source_id in selection.saved_sources:
            try:
                source_uuid = uuid.UUID(source_id)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Ungültige Quellen-ID: {source_id}")
            source = db.query(ResearchSource).filter(ResearchSource.id == source_uuid).first()
            if not source:
                raise HTTPException(status_code=404, detail=f"Quelle {source_id} wurde nicht gefunden")
            collected_sources.append(
                {
                    "id": str(source.id),
                    "title": source.title,
                    "url": source.url,
                    "description": source.description,
                    "document_type": source.document_type,
                    "download_path": source.download_path,
                    "created_at": source.created_at.isoformat() if source.created_at else None,
                }
            )
        collected["saved_sources"] = collected_sources

    return collected


def _build_generation_prompts(
    body,
    collected: Dict[str, List[Dict[str, Optional[str]]]],
    primary_bescheid_label: str,
    primary_bescheid_description: str,
    context_summary: str
) -> tuple[str, str]:
    """Build system and user prompts for document generation with strategic, flexible approach."""
    system_prompt = (
        f"Du bist ein erfahrener Fachanwalt für Migrationsrecht. Du schreibst eine überzeugende, "
        f"strategisch durchdachte juristische Argumentation gegen den Hauptbescheid (Anlage K2: {primary_bescheid_label}).\n\n"

        "STRATEGISCHER ANSATZ:\n"
        "Konzentriere dich auf die aussichtsreichsten Argumente. Nicht jeder Punkt des BAMF-Bescheids muss widerlegt werden - "
        "wähle die stärksten rechtlichen und tatsächlichen Ansatzpunkte aus den bereitgestellten Dokumenten.\n\n"

        "RECHTSGRUNDLAGEN:\n"
        "Stütze deine Argumentation auf die relevanten Vorschriften (§ 3 AsylG, § 4 AsylG, § 60 AufenthG etc.) "
        "und arbeite heraus, wo das BAMF diese fehlerhaft angewendet hat.\n\n"

        "BEWEISFÜHRUNG:\n"
        "- Hauptbescheid (Anlage K2): Zeige konkret, wo die Würdigung fehlerhaft ist (mit Seitenzahlen)\n"
        "- Anhörungen: Belege mit direkten Zitaten, was der Mandant tatsächlich ausgesagt hat (Bl. X d.A.)\n"
        "- Rechtsprechung: Zeige vergleichbare Fälle und übertragbare Rechtssätze\n"
        "- Gesetzestexte: Lege die Tatbestandsmerkmale zutreffend aus\n\n"

        "ZITIERWEISE:\n"
        "- Hauptbescheid: 'Anlage K2, S. X'\n"
        "- Anhörungen/Aktenbestandteile: 'Bl. X d.A.' oder 'Bl. X ff. d.A.'\n"
        "- Rechtsprechung: Volles Aktenzeichen, Gericht, Datum\n"
        "- Gesetzestexte: '§ X AsylG' bzw. '§ X Abs. Y AufenthG'\n\n"

        "STIL & FORMAT:\n"
        "- Durchgehender Fließtext ohne Aufzählungen oder Zwischenüberschriften\n"
        "- Klare Absatzstruktur: Einleitung, mehrere Argumentationsblöcke, Schluss\n"
        "- Jede Behauptung mit konkretem Beleg (Zitat, Fundstelle)\n"
        "- Präzise juristische Sprache, keine Floskeln\n"
        "- KEINE Antragsformulierung - nur die rechtliche Würdigung\n\n"

        "QUALITÄT VOR QUANTITÄT:\n"
        "Drei starke, gut belegte Argumente sind besser als zehn oberflächliche Punkte."
    )

    primary_bescheid_section = f"Hauptbescheid (Anlage K2): {primary_bescheid_label}"
    if primary_bescheid_description:
        primary_bescheid_section += f"\nBeschreibung: {primary_bescheid_description}"

    user_prompt = (
        f"Dokumententyp: {body.document_type}\n"
        f"{primary_bescheid_section}\n\n"

        f"Auftrag:\n{body.user_prompt.strip()}\n\n"

        "Verfügbare Dokumente:\n"
        f"{context_summary or '- (Keine Dokumente)'}\n\n"

        "VORGEHENSWEISE:\n"
        "1. Analysiere den Hauptbescheid (Anlage K2): Welche Ablehnungsgründe führt das BAMF an?\n"
        "2. Prüfe die Anhörungen und weitere Aktenbestandteile: Welche Aussagen widersprechen der BAMF-Würdigung?\n"
        "3. Prüfe die Rechtsprechung: Welche Urteile stützen die Position des Mandanten?\n"
        "4. Prüfe die Gesetzestexte: Welche Tatbestandsmerkmale sind erfüllt/nicht erfüllt?\n"
        "5. Wähle die 2-4 stärksten Argumente aus und entwickele diese detailliert.\n\n"

        "Verfasse nun eine überzeugende rechtliche Würdigung als Fließtext. Beginne erst nach gründlicher Analyse aller Dokumente."
    )

    return system_prompt, user_prompt


JLAWYER_BASE_URL = os.environ.get("JLAWYER_BASE_URL")
if JLAWYER_BASE_URL:
    JLAWYER_BASE_URL = JLAWYER_BASE_URL.rstrip("/")
JLAWYER_USERNAME = os.environ.get("JLAWYER_USERNAME")
JLAWYER_PASSWORD = os.environ.get("JLAWYER_PASSWORD")
JLAWYER_TEMPLATE_FOLDER_DEFAULT = os.environ.get("JLAWYER_TEMPLATE_FOLDER")
JLAWYER_PLACEHOLDER_KEY = os.environ.get("JLAWYER_PLACEHOLDER_KEY", "HAUPTTEXT")

def _sanitize_filename_for_claude(filename: str) -> str:
    """Sanitize filename to only contain ASCII characters for Claude Files API."""
    # Replace German umlauts and special characters
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'Ä': 'Ae', 'Ö': 'Oe', 'Ü': 'Ue'
    }
    for char, replacement in replacements.items():
        filename = filename.replace(char, replacement)

    # Remove any remaining non-ASCII characters and keep only safe chars
    sanitized = ''.join(
        c if c.isascii() and (c.isalnum() or c in '.-_ ') else '_'
        for c in filename
    )

    # Clean up multiple underscores/spaces and trim
    sanitized = re.sub(r'[_\s]+', '_', sanitized).strip('_')

    # Ensure we keep the .pdf extension
    if not sanitized.lower().endswith('.pdf'):
        sanitized += '.pdf'

    return sanitized


def _upload_documents_to_claude(client: anthropic.Anthropic, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, str]]:
    """Upload local documents using Claude Files API and return document content blocks.

    Prefers OCR'd text when available for better accuracy.
    """
    content_blocks: List[Dict[str, str]] = []
    MAX_PAGES = 100  # Claude Files API limit

    for entry in documents:
        original_filename = entry.get("filename") or "document"

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                print(f"[INFO] Verwende OCR-Text für {original_filename}")

            # Check PDF page count before uploading (only for PDFs)
            if mime_type == "application/pdf":
                try:
                    pdf = pikepdf.open(file_path)
                    page_count = len(pdf.pages)
                    pdf.close()

                    if page_count > MAX_PAGES:
                        print(f"[WARN] Datei {original_filename} hat {page_count} Seiten (max {MAX_PAGES}), wird übersprungen")
                        continue
                except Exception as exc:
                    print(f"[WARN] Seitenzahl für {original_filename} konnte nicht ermittelt werden: {exc}")

            # Sanitize filename for Claude API
            if mime_type == "text/plain":
                sanitized_filename = _sanitize_filename_for_claude(original_filename).replace('.pdf', '.txt')
            else:
                sanitized_filename = _sanitize_filename_for_claude(original_filename)

            # Upload file
            try:
                with open(file_path, "rb") as file_handle:
                    uploaded_file = client.beta.files.upload(
                        file=(sanitized_filename, file_handle, mime_type),
                        betas=["files-api-2025-04-14"],
                    )

                content_blocks.append({
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": uploaded_file.id,
                    },
                    "title": original_filename + (" (OCR)" if mime_type == "text/plain" else ""),
                })

            finally:
                # Clean up temporary file if needed
                if needs_cleanup:
                    try:
                        os.unlink(file_path)
                    except:
                        pass

        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARN] Überspringe {original_filename}: {exc}")
            continue
        except Exception as exc:
            print(f"[ERROR] Upload für {original_filename} fehlgeschlagen: {exc}")
            continue

    return content_blocks


def _upload_documents_to_openai(client: OpenAI, documents: List[Dict[str, Optional[str]]]) -> List[Dict[str, str]]:
    """Upload local documents using OpenAI Files API and return input_file blocks.

    Prefers OCR'd text when available for better accuracy.
    """
    file_blocks: List[Dict[str, str]] = []

    for entry in documents:
        original_filename = entry.get("filename") or "document"

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                print(f"[INFO] Using OCR text for {original_filename}")

            # Upload file
            try:
                with open(file_path, "rb") as file_handle:
                    uploaded_file = client.files.create(
                        file=file_handle,
                        purpose="user_data"
                    )

                file_blocks.append({
                    "type": "input_file",
                    "file_id": uploaded_file.id,
                })
                print(f"[DEBUG] Uploaded {original_filename} ({mime_type}) -> file_id: {uploaded_file.id}")

            finally:
                # Clean up temporary file if needed
                if needs_cleanup:
                    try:
                        os.unlink(file_path)
                    except:
                        pass

        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARN] Skipping {original_filename}: {exc}")
            continue
        except Exception as exc:
            print(f"[ERROR] OpenAI upload failed for {original_filename}: {exc}")
            continue

    return file_blocks








_CATEGORY_LABELS = {
    "anhoerung": "Anhörung",
    "bescheid": "Bescheid",
    "rechtsprechung": "Rechtsprechung",
    "saved_sources": "Gespeicherte Quelle",
}


def _normalize_for_match(value: Optional[str]) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^\w\s/.:,-]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_DATE_REGEX = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})")
_AZ_REGEX = re.compile(r"(?:Az\.?|Aktenzeichen)\s*[:]?\s*([A-Za-z0-9./\-\s]+)")


def _build_reference_candidates(category: str, entry: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    specific: List[str] = []
    generic: List[str] = []

    filename = entry.get("filename")
    if filename:
        stem = Path(filename).stem
        specific.extend(
            {
                filename,
                stem,
                stem.replace("_", " "),
                stem.replace("-", " "),
                stem.replace("_", "-"),
            }
        )

    title = entry.get("title")
    if title:
        specific.append(title)

    explanation = entry.get("explanation")
    if explanation:
        specific.append(explanation)
        specific.extend(_DATE_REGEX.findall(explanation))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(explanation)])
        for az in _AZ_REGEX.findall(explanation):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    url = entry.get("url")
    if url:
        specific.append(url)
        try:
            parsed = urlparse(url)
            if parsed.netloc:
                specific.append(parsed.netloc)
            if parsed.path:
                specific.append(parsed.path)
        except Exception:
            pass

    description = entry.get("description")
    if description:
        specific.append(description)
        specific.extend(_DATE_REGEX.findall(description))
        specific.extend([f"vom {m}" for m in _DATE_REGEX.findall(description)])
        for az in _AZ_REGEX.findall(description):
            cleaned = az.strip()
            if cleaned:
                specific.append(cleaned)
                specific.append(f"az {cleaned.lower()}")
                specific.append(f"az. {cleaned}")

    for key in ("date", "aktenzeichen"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            if key == "date":
                specific.append(value.strip())
                specific.append(f"vom {value.strip()}")
            else:
                specific.append(value.strip())
                specific.append(f"az {value.strip().lower()}")
                specific.append(f"az. {value.strip()}")

    category_generic = {
        "anhoerung": ["Anhörung"],
        "bescheid": ["Bescheid"],
        "rechtsprechung": ["Rechtsprechung", "Urteil"],
        "saved_sources": ["Quelle", "Research"],
    }
    generic.extend(category_generic.get(category, []))

    def _dedupe(items):
        seen = set()
        result = []
        for item in items:
            if not item:
                continue
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    return {"specific": _dedupe(specific), "generic": _dedupe(generic)}


def _generate_with_claude(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: str,
    document_blocks: List[Dict]
) -> str:
    """Call Claude API and return generated text."""
    content = [{"type": "text", "text": user_prompt}]
    content.extend(document_blocks)

    print(f"[DEBUG] Content blocks being sent to Claude API:")
    for i, block in enumerate(content):
        block_type = block.get("type")
        if block_type == "text":
            print(f"  [{i}] text block (length: {len(block.get('text', ''))})")
        elif block_type == "document":
            print(f"  [{i}] document block: {block.get('title', 'untitled')} (file_id: {block.get('source', {}).get('file_id', 'N/A')})")
        else:
            print(f"  [{i}] {block_type} block")

    response = client.beta.messages.create(
        model="claude-sonnet-4-5",
        system=system_prompt,
        max_tokens=12288,
        messages=[{"role": "user", "content": content}],
        temperature=0.2,
        betas=["files-api-2025-04-14"],
    )

    # Log API response metadata
    stop_reason = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    if stop_reason is not None:
        print(f"[DEBUG] API Response - stop_reason: {stop_reason}")
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        print(f"[DEBUG] API Response - usage: input={input_tokens}, output={output_tokens}")

    # Extract text from response blocks
    text_parts: List[str] = []
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        else:
            block_type = getattr(block, "type", None)
            block_text = getattr(block, "text", None)
            if block_type == "text" and block_text:
                text_parts.append(block_text)

    generated_text = "\n\n".join([part for part in text_parts if part]).strip()

    # Warn if we hit max_tokens limit
    if stop_reason == "max_tokens":
        print("[WARN] Generation stopped due to max_tokens limit - output may be incomplete!")
        if usage is not None:
            print(f"[WARN] Consider increasing max_tokens above {getattr(usage, 'output_tokens', 'unknown')}")

    return generated_text


def _generate_with_gpt5(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    file_blocks: List[Dict],
    reasoning_effort: str = "high",
    verbosity: str = "high",
    model: str = "gpt-5.1"
) -> str:
    """Call GPT-5 Responses API and return generated text.

    Uses OpenAI Responses API with:
    - Reasoning effort: configurable (minimal/low/medium/high)
    - Output verbosity: configurable (low/medium/high)
    - Max output tokens: 12288 (comprehensive legal briefs)
    """

    # Build input array with system and user messages
    input_messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                *file_blocks,  # Files BEFORE text (important for context)
                {"type": "input_text", "text": user_prompt}
            ]
        }
    ]

    print(f"[DEBUG] Calling GPT-5.1 Responses API:")
    print(f"  Model: {model}")
    print(f"  Files: {len(file_blocks)}")
    print(f"  Reasoning effort: {reasoning_effort}")
    print(f"  Output verbosity: {verbosity}")

    # Responses API call
    # Note: temperature, top_p, logprobs NOT supported for GPT-5!
    response = client.responses.create(
        model=model,
        input=input_messages,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
        max_output_tokens=12288,
    )

    # Extract text from response
    # Method 1: Direct property (if available)
    if hasattr(response, 'output_text') and response.output_text:
        generated_text = response.output_text
    # Method 2: Navigate output structure
    elif response.output and len(response.output) > 0:
        message = response.output[0]
        if hasattr(message, 'content') and message.content and len(message.content) > 0:
            content_block = message.content[0]
            generated_text = content_block.text if hasattr(content_block, 'text') else ""
        else:
            generated_text = ""
    else:
        generated_text = ""

    # Log response metadata
    if hasattr(response, 'status'):
        print(f"[DEBUG] GPT-5 Response - status: {response.status}")

    if hasattr(response, 'usage'):
        usage = response.usage
        print(f"[DEBUG] GPT-5 Response - tokens: input={usage.input_tokens}, output={usage.output_tokens}, total={usage.total_tokens}")

        # Log reasoning tokens (unique to reasoning models)
        if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
            if reasoning_tokens > 0:
                print(f"[DEBUG] GPT-5 Reasoning tokens: {reasoning_tokens} ({reasoning_tokens/usage.output_tokens*100:.1f}% of output)")

    # Check if response was completed
    if hasattr(response, 'status') and response.status != 'completed':
        print(f"[WARN] GPT-5 response status: {response.status}")
        if hasattr(response, 'incomplete_details') and response.incomplete_details:
            print(f"[WARN] Incomplete details: {response.incomplete_details}")

    return generated_text.strip()


def _upload_documents_to_gemini(client: genai.Client, documents: List[Dict[str, Optional[str]]]) -> List[types.File]:
    """Upload local documents using Gemini Files API and return file objects.

    Prefers OCR'd text when available for better accuracy.
    """
    uploaded_files: List[types.File] = []

    for entry in documents:
        original_filename = entry.get("filename") or "document"

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)
            temp_text_path = file_path if needs_cleanup else None

            if mime_type == "text/plain":
                print(f"[INFO] Using OCR text for {original_filename}")

            # Upload file
            try:
                with open(file_path, "rb") as file_handle:
                    uploaded_file = client.files.upload(
                        file=file_handle,
                        config={
                            "mime_type": mime_type,
                            "display_name": original_filename
                        }
                    )

                print(f"[DEBUG] Uploaded {original_filename} ({mime_type}) -> uri: {uploaded_file.uri}")

                # Wait for file to be active if it's a PDF (Gemini needs processing time)
                if mime_type == "application/pdf":
                    import time
                    print(f"[DEBUG] Waiting for PDF processing: {original_filename}")
                    while uploaded_file.state.name == "PROCESSING":
                        time.sleep(1)
                        uploaded_file = client.files.get(name=uploaded_file.name)
                    
                    if uploaded_file.state.name != "ACTIVE":
                        print(f"[WARN] File {original_filename} is in state {uploaded_file.state.name}, might fail")

                uploaded_files.append(uploaded_file)

            finally:
                # Clean up temporary file if needed
                if needs_cleanup:
                    try:
                        os.unlink(file_path)
                    except:
                        pass

        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARN] Skipping {original_filename}: {exc}")
            continue
        except Exception as exc:
            print(f"[ERROR] Gemini upload failed for {original_filename}: {exc}")
            continue

    return uploaded_files


def _generate_with_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    files: List[types.File],
    model: str = "gemini-3-pro-preview"
) -> str:
    """Call Gemini API and return generated text."""
    
    print(f"[DEBUG] Calling Gemini API:")
    print(f"  Model: {model}")
    print(f"  Files: {len(files)}")

    # Combine user prompt and files
    # The SDK accepts a list of [prompt, file1, file2, ...]
    contents = []
    
    # Add files first (context)
    if files:
        contents.extend(files)
    
    # Add user prompt
    contents.append(user_prompt)

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=1.0,
                max_output_tokens=12288,
            )
        )
        
        # Log usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            print(f"[DEBUG] Gemini Response - tokens: input={usage.prompt_token_count}, output={usage.candidates_token_count}")

        return response.text or ""

    except Exception as e:
        print(f"[ERROR] Gemini generation failed: {e}")
        # Print more details if available
        if hasattr(e, 'details'):
            print(f"Details: {e.details}")
        raise






def verify_citations(
    generated_text: str,
    selected_documents: Dict[str, List[Dict[str, Optional[str]]]],
    sources_metadata: Optional[Dict[str, List[Dict[str, Optional[str]]]]] = None,
) -> Dict[str, List[str]]:
    """
    Verify that selected sources appear in the generated text.
    Returns dict with keys `cited`, `missing`, `warnings`.
    """
    normalized_text = _normalize_for_match(generated_text or "")
    result: Dict[str, List[str]] = {"cited": [], "missing": [], "warnings": []}

    if not normalized_text:
        result["warnings"].append("Generierter Text ist leer; Zitatprüfung nicht möglich.")
        for category, entries in (selected_documents or {}).items():
            category_label = _CATEGORY_LABELS.get(category, category)
            for entry in entries:
                label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
                result["missing"].append(f"{category_label}: {label}")
        return result

    seen_labels: set[str] = set()

    for category, entries in (selected_documents or {}).items():
        category_label = _CATEGORY_LABELS.get(category, category)
        for entry in entries:
            base_label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
            label = f"{category_label}: {base_label}"

            if label in seen_labels:
                continue
            seen_labels.add(label)

            candidates = _build_reference_candidates(category, entry)
            match_found = False
            generic_hit = False

            for candidate in candidates["specific"]:
                normalized_candidate = _normalize_for_match(candidate)
                if normalized_candidate and normalized_candidate in normalized_text:
                    match_found = True
                    break

            if not match_found:
                for candidate in candidates["generic"]:
                    normalized_candidate = _normalize_for_match(candidate)
                    if normalized_candidate and normalized_candidate in normalized_text:
                        match_found = True
                        generic_hit = True
                        break

            if match_found:
                result["cited"].append(label)
                if generic_hit:
                    result["warnings"].append(
                        f"{label}: nur generischer Hinweis gefunden – bitte Zitierung prüfen."
                    )
            else:
                result["missing"].append(label)

            if entry.get("role") == "primary":
                if "anlage k2" not in normalized_text:
                    result["warnings"].append(
                        f"{label}: Referenz 'Anlage K2' nicht gefunden – bitte kontrollieren."
                    )

    return result


def _is_jlawyer_configured() -> bool:
    return all([
        JLAWYER_BASE_URL,
        JLAWYER_USERNAME,
        JLAWYER_PASSWORD,
        JLAWYER_PLACEHOLDER_KEY,
    ])


@router.get("/jlawyer/templates", response_model=JLawyerTemplatesResponse)
@limiter.limit("20/hour")
async def get_jlawyer_templates(request: Request, folder: Optional[str] = None):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    folder_name = (folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    url = f"{JLAWYER_BASE_URL}/v6/templates/documents/{quote(folder_name, safe='')}"
    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, auth=auth)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    templates: List[str] = []
    try:
        payload = response.json()
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, str):
                    templates.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("template") or item.get("fileName")
                    if isinstance(name, str):
                        templates.append(name)
    except ValueError:
        text = response.text or ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                templates.append(line)

    return JLawyerTemplatesResponse(templates=templates, folder=folder_name)


@router.post("/generate", response_model=GenerationResponse)
@limiter.limit("10/hour")
async def generate(request: Request, body: GenerationRequest, db: Session = Depends(get_db)):
    """Generate drafts using Claude Sonnet 4.5 or GPT-5 Reasoning models."""

    # 1. Collect documents (shared logic)
    collected = _collect_selected_documents(body.selected_documents, db)

    # 2. Build document list (shared logic)
    document_entries: List[Dict[str, Optional[str]]] = []
    for category in ("anhoerung", "bescheid", "rechtsprechung"):
        document_entries.extend(collected.get(category, []))

    for source_entry in collected.get("saved_sources", []):
        download_path = source_entry.get("download_path")
        if not download_path:
            continue
        document_entries.append({
            "filename": source_entry.get("title") or source_entry.get("id"),
            "file_path": download_path,
            "category": source_entry.get("document_type") or "Quelle",
        })

    print(f"[DEBUG] Collected {len(document_entries)} document entries for upload")
    for entry in document_entries:
        print(f"  - {entry.get('category', 'N/A')}: {entry.get('filename', 'N/A')}")

    # 3. Build prompts (shared logic)
    context_summary = _summarize_selection_for_prompt(collected)

    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    primary_bescheid_label = (
        (primary_bescheid_entry.get("filename") or "—") if primary_bescheid_entry else "—"
    )
    primary_bescheid_description = ""
    if primary_bescheid_entry:
        explanation = primary_bescheid_entry.get("explanation")
        if explanation:
            primary_bescheid_description = explanation.strip()

    print(f"[DEBUG] Context summary:\n{context_summary}")
    if primary_bescheid_entry:
        print(f"[DEBUG] Primary Bescheid identified: {primary_bescheid_label}")

    system_prompt, user_prompt = _build_generation_prompts(
        body, collected, primary_bescheid_label,
        primary_bescheid_description, context_summary
    )

    # 4. Route to provider-specific logic
    try:
        if body.model.startswith("gpt"):
            # GPT-5.1 path (Responses API)
            print(f"[DEBUG] Using OpenAI GPT-5.1: {body.model}")
            client = get_openai_client()
            file_blocks = _upload_documents_to_openai(client, document_entries)
            print(f"[DEBUG] Uploaded {len(file_blocks)} documents to OpenAI Files API")

            generated_text = _generate_with_gpt5(
                client, system_prompt, user_prompt, file_blocks,
                reasoning_effort="high",
                verbosity=body.verbosity,
                model=body.model
            )
        elif body.model.startswith("gemini"):
            # Gemini path
            print(f"[DEBUG] Using Google Gemini: {body.model}")
            client = get_gemini_client()
            files = _upload_documents_to_gemini(client, document_entries)
            print(f"[DEBUG] Uploaded {len(files)} documents to Gemini Files API")

            generated_text = _generate_with_gemini(
                client, system_prompt, user_prompt, files,
                model=body.model
            )
        else:
            # Claude path (default)
            print(f"[DEBUG] Using Anthropic Claude: {body.model}")
            client = get_anthropic_client()
            document_blocks = _upload_documents_to_claude(client, document_entries)
            print(f"[DEBUG] Uploaded {len(document_blocks)} documents to Claude Files API")

            generated_text = _generate_with_claude(
                client, system_prompt, user_prompt, document_blocks
            )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {exc}")

    citations = verify_citations(generated_text, collected)
    if citations.get("warnings"):
        for warning in citations["warnings"]:
            print(f"[CITATION WARNING] {warning}")
    if citations.get("missing"):
        for missing in citations["missing"]:
            print(f"[CITATION MISSING] {missing}")
    metadata = GenerationMetadata(
        documents_used={
            "anhoerung": len(collected.get("anhoerung", [])),
            "bescheid": len(collected.get("bescheid", [])),
            "rechtsprechung": len(collected.get("rechtsprechung", [])),
            "saved_sources": len(collected.get("saved_sources", [])),
        },
        citations_found=len(citations.get("cited", [])),
        missing_citations=citations.get("missing", []),
        warnings=citations.get("warnings", []),
        word_count=len(generated_text.split()) if generated_text else 0,
    )

    structured_used_documents: List[Dict[str, str]] = []
    for category, entries in collected.items():
        for entry in entries:
            filename = entry.get("filename") or entry.get("title")
            if not filename:
                continue
            payload = {"filename": filename, "category": category}
            role = entry.get("role")
            if role:
                payload["role"] = role
            structured_used_documents.append(payload)

    return GenerationResponse(
        document_type=body.document_type,
        user_prompt=body.user_prompt.strip(),
        generated_text=generated_text or "(Kein Text erzeugt)",
        used_documents=structured_used_documents,
        metadata=metadata,
    )


def _summarize_selection_for_prompt(collected: Dict[str, List[Dict[str, Optional[str]]]]) -> str:
    """Create a short textual summary of the selected sources for the Claude prompt."""
    lines: List[str] = []
    primary_entry = None
    for entry in collected.get("bescheid", []):
        if entry.get("role") == "primary":
            primary_entry = entry
            break

    if primary_entry:
        lines.append(f"- Hauptbescheid (Anlage K2): {primary_entry.get('filename')}")
        primary_explanation = (primary_entry.get("explanation") or "").strip()
        if primary_explanation:
            lines.append(f"  Beschreibung: {primary_explanation}")

    def _append_section(header: str, entries: List[Dict[str, Optional[str]]], note: Optional[str] = None) -> None:
        if not entries:
            return
        lines.append(f"\n{header}")
        if note:
            lines.append(note)
        for entry in entries:
            label = entry.get("filename") or entry.get("title") or entry.get("id") or "Unbekanntes Dokument"
            explanation = (entry.get("explanation") or "").strip()
            if explanation:
                lines.append(f"- {label} — {explanation}")
            else:
                lines.append(f"- {label}")

    _append_section(
        "📋 Anhörungen:",
        collected.get("anhoerung", []),
        "Bitte als 'Bl. ... der Akte' zitieren.",
    )

    other_bescheide = [e for e in collected.get("bescheid", []) if e.get("role") != "primary"]
    _append_section(
        "📄 Weitere Bescheide / Aktenauszüge:",
        other_bescheide,
        "Bitte als 'Bl. ... der Akte' zitieren.",
    )

    _append_section(
        "⚖️ Rechtsprechung:",
        collected.get("rechtsprechung", []),
    )

    saved_sources = collected.get("saved_sources", [])
    if saved_sources:
        lines.append("\n🔗 Gespeicherte Quellen:")
        for entry in saved_sources:
            title = entry.get("title") or entry.get("id") or "Unbekannte Quelle"
            url = entry.get("url") or "keine URL"
            description = (entry.get("description") or "").strip()
            base_line = f"- {title} ({url})"
            if description:
                lines.append(f"{base_line} — {description}")
            else:
                lines.append(base_line)

    return "\n".join(lines)


@router.post("/send-to-jlawyer", response_model=JLawyerResponse)
@limiter.limit("10/hour")
async def send_to_jlawyer(request: Request, body: JLawyerSendRequest):
    if not _is_jlawyer_configured():
        raise HTTPException(status_code=503, detail="j-lawyer Integration ist nicht konfiguriert")

    case_id = body.case_id.strip()
    template_name = body.template_name.strip()
    file_name = body.file_name.strip()
    template_folder = (body.template_folder or JLAWYER_TEMPLATE_FOLDER_DEFAULT or "").strip()

    if not case_id or not template_name or not file_name:
        raise HTTPException(status_code=400, detail="case_id, template_name und file_name sind Pflichtfelder")

    if not template_folder:
        raise HTTPException(status_code=400, detail="Kein Template-Ordner konfiguriert")

    if not file_name.lower().endswith(".odt"):
        file_name = f"{file_name}.odt"

    placeholder_value = body.generated_text or ""

    url = (
        f"{JLAWYER_BASE_URL}/v6/templates/documents/"
        f"{quote(template_folder, safe='')}/"
        f"{quote(template_name, safe='')}/"
        f"{quote(case_id, safe='')}/"
        f"{quote(file_name, safe='')}"
    )

    payload = [
        {
            "placeHolderKey": JLAWYER_PLACEHOLDER_KEY,
            "placeHolderValue": placeholder_value,
        }
    ]

    auth = (JLAWYER_USERNAME, JLAWYER_PASSWORD)

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(url, auth=auth, json=payload)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"j-lawyer Anfrage fehlgeschlagen: {exc}")

    if response.status_code >= 400:
        detail = response.text or response.reason_phrase or "Unbekannter Fehler"
        raise HTTPException(status_code=502, detail=f"j-lawyer Fehler ({response.status_code}): {detail}")

    return JLawyerResponse(success=True, message="Vorlage erfolgreich an j-lawyer gesendet")


