import json
import re
import unicodedata
import uuid
from pathlib import Path
from pydantic import BaseModel, Field, create_model
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
        "anonymization_metadata": doc.anonymization_metadata,
        "is_anonymized": doc.is_anonymized,
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
    if selection.vorinstanz.primary:
        total_selected += 1
    total_selected += len(selection.vorinstanz.others)
    total_selected += 1 if bescheid_selection.primary else 0
    total_selected += len(bescheid_selection.others)

    if total_selected == 0:
        raise HTTPException(status_code=400, detail="Bitte wählen Sie mindestens ein Dokument aus")

    collected: Dict[str, List[Dict[str, Optional[str]]]] = {
        "anhoerung": [],
        "bescheid": [],
        "rechtsprechung": [],
        "anhoerung": [],
        "bescheid": [],
        "vorinstanz": [],
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

    # Load Vorinstanz documents
    vorinstanz_selection = selection.vorinstanz
    vorinstanz_filenames = []
    if vorinstanz_selection.primary:
        vorinstanz_filenames.append(vorinstanz_selection.primary)
    if vorinstanz_selection.others:
        vorinstanz_filenames.extend(vorinstanz_selection.others)

    if vorinstanz_filenames:
        query = (
            db.query(Document)
            .filter(Document.filename.in_(vorinstanz_filenames))
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in vorinstanz_filenames if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Vorinstanz-Dokumente nicht gefunden: {', '.join(missing)}")
        
        if vorinstanz_selection.primary:
            primary_doc = found_map.get(vorinstanz_selection.primary)
            if primary_doc:
                _validate_category(primary_doc, DocumentCategory.VORINSTANZ.value)
                doc_dict = _document_to_context_dict(primary_doc)
                content_len = len(doc_dict.get("content") or "")
                print(f"[DEBUG] Collected Primary Vorinstanz: {primary_doc.filename}, Content Length: {content_len}")
                collected["vorinstanz"].append({**doc_dict, "role": "primary"})

        for other_name in vorinstanz_selection.others or []:
            doc = found_map.get(other_name)
            if doc:
                _validate_category(doc, DocumentCategory.VORINSTANZ.value)
                doc_dict = _document_to_context_dict(doc)
                content_len = len(doc_dict.get("content") or "")
                print(f"[DEBUG] Collected Secondary Vorinstanz: {doc.filename}, Content Length: {content_len}")
                collected["vorinstanz"].append({**doc_dict, "role": "secondary"})

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
                    "category": "saved_source",
                    "download_path": source.download_path,
                    "created_at": source.created_at.isoformat() if source.created_at else None,
                    "gemini_file_uri": source.gemini_file_uri,
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
    
    # Check if this is an AZB (Appeal) scenario based on Vorinstanz documents
    is_azb = len(collected.get("vorinstanz", [])) > 0
    
    if is_azb:
        # --- AZB PROMPT LOGIC ---
        print("[DEBUG] AZB Mode activated (Vorinstanz documents present)")
        
        # Identify primary Vorinstanz document (Judgment)
        primary_vorinstanz_doc = next((d for d in collected.get("vorinstanz", []) if d.get("role") == "primary"), None)
        primary_vorinstanz_label = primary_vorinstanz_doc["filename"] if primary_vorinstanz_doc else "das Urteil"

        system_prompt = (
            "Du bist ein erfahrener Fachanwalt für Migrationsrecht, spezialisiert auf das Berufungszulassungsrecht. "
            f"Du schreibst eine Begründung für einen Antrag auf Zulassung der Berufung (AZB) gegen ein Urteil des Verwaltungsgerichts ({primary_vorinstanz_label}) in einer Asylstreitigkeit.\n\n"

            "WICHTIGE RECHTSLAGE (§ 78 Abs. 3 AsylG):\n"
            "In Asylverfahren gibt es den Zulassungsgrund der 'ernstlichen Zweifel' (§ 124 Abs. 2 Nr. 1 VwGO) NICHT. "
            "Die Berufung ist nur zuzulassen bei:\n"
            "1. Grundsätzlicher Bedeutung (§ 78 Abs. 3 Nr. 1 AsylG)\n"
            "2. Divergenz (§ 78 Abs. 3 Nr. 2 AsylG)\n"
            "3. Verfahrensmangel (§ 78 Abs. 3 Nr. 3 AsylG)\n\n"

            "ZIEL & FOKUS:\n"
            "Der Fokus liegt auf der Rüge von VERFAHRENSMÄNGELN, insbesondere der Verletzung des rechtlichen Gehörs (§ 78 Abs. 3 Nr. 3 AsylG i.V.m. § 138 Nr. 3 VwGO / Art. 103 Abs. 1 GG).\n\n"

            "STRATEGIE (GEHÖRSRÜGE):\n"
            f"1. Darlegung des übergangenen Vortrags: Welches konkrete Vorbringen oder Beweisangebot hat das Gericht in {primary_vorinstanz_label} ignoriert?\n"
            "2. Darlegung der Entscheidungserheblichkeit (Kausalität): Warum hätte das Gericht anders entschieden, wenn es den Vortrag berücksichtigt hätte?\n"
            "   - Argumentiere NICHT mit 'falscher Rechtsanwendung' (das wäre ein materieller Fehler), sondern mit 'Nichtzurkenntnisnahme von Vortrag' (Verfahrensfehler).\n\n"

            "WEITERE ZULASSUNGSGRÜNDE (NUR WENN EINSCHLÄGIG):\n"
            "- Grundsätzliche Bedeutung: Wenn eine klärungsbedürftige Rechts- oder Tatsachenfrage vorliegt, die über den Einzelfall hinausgeht.\n"
            "- Divergenz: Wenn das Urteil von einer Entscheidung des OVG, BVerwG oder BVerfG abweicht (genaue Bezeichnung der Abweichung erforderlich).\n\n"

            "FORMAT:\n"
            "- Juristischer Schriftsatzstil.\n"
            "- Keine Floskeln.\n"
            f"- Konkrete Bezugnahme auf das VG-Urteil ({primary_vorinstanz_label}).\n"
            "- Beginne direkt mit der Begründung, keine Adresszeilen."
        )
        
        # Verbosity for AZB
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += "\n\nFASSUNG (LOW): Fasse dich kurz. Nur die stärkste Gehörsrüge ausführen."
        elif verbosity == "medium":
            system_prompt += "\n\nFASSUNG (MEDIUM): Ausgewogene Begründung. Fokus auf die Gehörsverletzung."
        else: # high
            system_prompt += (
                "\n\nFASSUNG (HIGH):\n"
                "- Ausführliche und tiefe Auseinandersetzung mit dem Verfahrensfehler.\n"
                "- Nutze die volle Token-Kapazität für eine erschöpfende Begründung.\n"
                "- Arbeite die Kausalität des Fehlers für das Urteil detailliert heraus."
            )

        user_prompt = (
            f"Dokumententyp: Antrag auf Zulassung der Berufung (AZB) nach § 78 AsylG\n\n"
            
            f"AUFTRAG:\n{body.user_prompt.strip()}\n\n"

            "VERFÜGBARE DOKUMENTE:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "VORGEHENSWEISE:\n"
            f"1. Analysiere das Urteil der Vorinstanz ({primary_vorinstanz_label}): Wo hat das Gericht Vortrag des Klägers ignoriert oder übergangen?\n"
            "2. Prüfe die Aktenlage (Anhörungen, Schriftsätze): Was wurde vorgetragen, aber im Urteil nicht erwähnt?\n"
            "3. Formuliere die Rüge der 'Verletzung des rechtlichen Gehörs' (§ 78 Abs. 3 Nr. 3 AsylG):\n"
            "   - 'Das Gericht hat den Anspruch des Klägers auf rechtliches Gehör verletzt (Art. 103 Abs. 1 GG), indem es...'\n"
            "   - 'Es hat folgenden wesentlichen Vortrag übergangen: ...'\n"
            "4. Begründe die Kausalität: 'Das Urteil beruht auf diesem Verfahrensmangel, denn es ist nicht auszuschließen, dass das Gericht bei Berücksichtigung des Vortrags zu einer anderen Entscheidung gelangt wäre.'\n"
            "5. Falls einschlägig: Prüfe Grundsätzliche Bedeutung oder Divergenz.\n\n"

            "Erstelle nun die Begründung des Zulassungsantrags als Fließtext. Vermeide Rügen der materiellen Rechtslage ('Ernstliche Zweifel'), da diese im Asylrecht unzulässig sind."
        )
        
    else:
        # --- STANDARD PROMPT LOGIC ---
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
            "- Vorinstanz: Gehe auf Urteile oder Protokolle der Vorinstanz ein, falls vorhanden\n"
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
            "- Beginne ohne Vorbemerkungen direkt mit dem juristischen Fließtext, keine Adresszeilen oder Anreden\n"
            "- KEINE Antragsformulierung - nur die rechtliche Würdigung\n\n"
    
            "Drei starke, gut belegte Argumente sind besser als zehn oberflächliche Punkte. Aber diese drei Argumente müssen erschöpfend behandelt werden."
        )
    
        # Adjust instructions based on verbosity
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: LOW):\n"
                "- Fasse dich kurz und prägnante.\n"
                "- Konzentriere dich ausschließlich auf die absolut wesentlichen Punkte.\n"
                "- Vermeide ausschweifende Erklärungen oder Wiederholungen.\n"
                "- Ziel ist eine kompakte, schnell erfassbare Argumentation."
            )
        elif verbosity == "medium":
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: MEDIUM):\n"
                "- Wähle einen ausgewogenen Ansatz zwischen Detailtiefe und Lesbarkeit.\n"
                "- Erkläre die wichtigen Punkte gründlich, aber komme schnell zum Punkt.\n"
                "- Vermeide unnötige Füllwörter."
            )
        else:  # high (default)
            system_prompt += (
                "\n\n"
                "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
                "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
                "- Nutze die volle verfügbare Länge (bis zu 12.000 Token), um den Sachverhalt umfassend zu würdigen.\n"
                "- Gehe detailliert auf jeden Widerspruch und jedes Beweismittel ein."
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
            "1. Analysiere den Hauptbescheid (Anlage K2): Welche Ablehnungsgründe führt das BAMF an? Zerlege die Argumentation des BAMF Schritt für Schritt.\n"
            "2. Prüfe die Anhörungen und weitere Aktenbestandteile: Welche Aussagen widersprechen der BAMF-Würdigung? Zitiere ausführlich.\n"
            "3. Prüfe Dokumente der Vorinstanz: Gibt es relevante Feststellungen aus früheren Verfahren?\n"
            "4. Prüfe die Rechtsprechung: Welche Urteile stützen die Position des Mandanten? Arbeite die Parallelen heraus.\n"
            "5. Prüfe die Gesetzestexte: Welche Tatbestandsmerkmale sind erfüllt/nicht erfüllt?\n"
            "6. Wähle die 2-4 stärksten Argumente aus und entwickele diese detailliert und umfassend.\n\n"
    
            "Verfasse nun eine überzeugende, detaillierte rechtliche Würdigung als Fließtext."
            "Beginne im ersten Satz unmittelbar mit der juristischen Argumentation ohne Adressblock, Anrede oder Meta-Hinweise."
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
    
    IMPORTANT: GPT-5.1 Responses API currently rejects .txt file uploads with "unsupported file type".
    Therefore, for text/plain content (OCR/Anonymized), we embed it DIRECTLY as an input_text block.
    We only upload actual PDFs.
    """
    file_blocks: List[Dict[str, str]] = []

    for entry in documents:
        original_filename = entry.get("filename") or "document"

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                print(f"[INFO] Embedding text content for {original_filename} (skipping upload)")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Create a text block with the document content
                    # We wrap it with a header to identify the document
                    text_block = f"DOKUMENT: {original_filename}\n\n{content}"
                    
                    file_blocks.append({
                        "type": "input_text",
                        "text": text_block,
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to read text file {file_path}: {e}")
            
            else:
                # It's a PDF (or other supported binary), upload it
                print(f"[INFO] Uploading PDF for {original_filename}")
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
                except Exception as exc:
                    print(f"[ERROR] OpenAI upload failed for {original_filename}: {exc}")

        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARN] Skipping {original_filename}: {exc}")
            continue
        finally:
            # Clean up temporary file if needed
            if 'needs_cleanup' in locals() and needs_cleanup:
                try:
                    os.unlink(file_path)
                except:
                    pass

    return file_blocks








_CATEGORY_LABELS = {
    "anhoerung": "Anhörung",
    "bescheid": "Bescheid",
    "vorinstanz": "Vorinstanz",
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
        "vorinstanz": ["Vorinstanz", "Urteil", "Protokoll"],
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
    user_prompt: Optional[str],
    document_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = []
) -> tuple[str, int]:
    """Call Claude API and return generated text."""
    
    messages = []

    if chat_history:
        # 1. First message: Files + Initial Prompt
        first_msg = chat_history[0]
        first_content = []
        # Add documents to the first message
        first_content.extend(document_blocks)
        first_content.append({"type": "text", "text": first_msg.get("content", "")})
        messages.append({"role": "user", "content": first_content})

        # 2. Subsequent messages
        for msg in chat_history[1:]:
            role = msg.get("role")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
        
        # 3. Current prompt
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})
            
    else:
        # Standard single-turn
        content = []
        if user_prompt:
            content.append({"type": "text", "text": user_prompt})
        content.extend(document_blocks)
        
        if not content:
             return "", 0
             
        messages.append({"role": "user", "content": content})

    # Debug log
    print(f"[DEBUG] Claude Messages: {len(messages)} turns")

    response = client.beta.messages.create(
        model="claude-opus-4-5",
        system=system_prompt,
        max_tokens=12288,
        messages=messages,
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

    total_tokens = 0
    if usage:
        total_tokens = (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "output_tokens", 0) or 0)

    return generated_text, total_tokens


def _generate_with_gpt5(
    client: OpenAI,
    system_prompt: str,
    user_prompt: Optional[str],
    file_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = [],
    reasoning_effort: str = "high",
    verbosity: str = "high",
    model: str = "gpt-5.1"
) -> tuple[str, int]:
    """Call GPT-5 Responses API and return generated text.

    Uses OpenAI Responses API with:
    - Reasoning effort: configurable (minimal/low/medium/high)
    - Output verbosity: configurable (low/medium/high)
    - Max output tokens: 12288 (comprehensive legal briefs)
    """

    # Build initial system message
    input_messages = [
        {
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt}
            ]
        }
    ]

    # If we have history, we need to reconstruct the conversation
    # The first user message MUST contain the files
    if chat_history:
        # 1. First user message (Files + Initial Prompt)
        first_user_msg = chat_history[0]
        input_messages.append({
            "role": "user",
            "content": [
                *file_blocks,
                {"type": "input_text", "text": first_user_msg.get("content", "")}
            ]
        })

        # 2. Subsequent messages (Assistant replies, User follow-ups)
        # OPTIMIZATION: For 30k TPM limit, we truncate the middle of the history
        # We keep the last 4 messages (2 turns) to maintain immediate context
        remaining_history = chat_history[1:]
        if len(remaining_history) > 4:
            print(f"[DEBUG] Truncating history from {len(remaining_history)} to last 4 messages to save tokens.")
            remaining_history = remaining_history[-4:]

        for msg in remaining_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                input_messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": content}]
                })
            elif role == "assistant":
                input_messages.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}]
                })
        
        # 3. The CURRENT user prompt (if not already in history)
        if user_prompt and (not chat_history or chat_history[-1].get("content") != user_prompt):
             input_messages.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}]
            })

    else:
        # No history - Standard single-turn request
        input_messages.append({
            "role": "user",
            "content": [
                *file_blocks,  # Files BEFORE text
                {"type": "input_text", "text": user_prompt}
            ]
        })

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

    total_tokens = 0
    if hasattr(response, 'usage'):
        usage = response.usage
        total_tokens = getattr(usage, 'total_tokens', 0)

    return generated_text.strip(), total_tokens


def _upload_documents_to_gemini(client: genai.Client, documents: List[Dict[str, Optional[str]]]) -> List[types.File]:
    """Upload local documents using Gemini Files API and return file objects.
    
    Checks database first for existing Gemini URIs to avoid re-uploading.
    Prefers OCR'd text when available for better accuracy.
    """
    uploaded_files: List[types.File] = []
    
    # We need a DB session to check for existing URIs
    # Since this is a helper, we'll create a local session or pass it in.
    # For simplicity/safety in this async context, let's assume we can get a session.
    from database import SessionLocal
    db = SessionLocal()

    try:
        for entry in documents:
            original_filename = entry.get("filename")
            doc_id = entry.get("id")
            
            # 1. Check DB for existing URI
            existing_uri = None
            if doc_id:
                try:
                    doc_uuid = uuid.UUID(doc_id)
                    db_doc = db.query(Document).filter(Document.id == doc_uuid).first()
                    if db_doc and db_doc.gemini_file_uri:
                        existing_uri = db_doc.gemini_file_uri
                        print(f"[DEBUG] Found existing Gemini URI for {original_filename}: {existing_uri}")
                except Exception as e:
                    print(f"[WARN] Failed to check DB for Gemini URI: {e}")

            if existing_uri:
                try:
                    # Extract file name from URI (e.g., https://generativelanguage.googleapis.com/v1beta/files/abc123xyz -> files/abc123xyz)
                    # The client.files.get() expects 'files/...' format or just the ID? 
                    # Actually, the Python SDK `client.files.get(name=...)` expects 'files/ID'.
                    
                    file_name = None
                    if "/files/" in existing_uri:
                        file_name = "files/" + existing_uri.split("/files/")[-1]
                    
                    if file_name:
                        # Check if file still exists and is valid
                        existing_file = client.files.get(name=file_name)
                        if existing_file.state.name == "ACTIVE":
                            print(f"[DEBUG] Reusing existing Gemini file for {original_filename}: {file_name}")
                            uploaded_files.append(existing_file)
                            continue
                        else:
                            print(f"[DEBUG] Existing file {file_name} is not ACTIVE (State: {existing_file.state.name}), re-uploading.")
                    else:
                        print(f"[WARN] Could not extract name from URI: {existing_uri}")
                except Exception as e:
                    # Catch 403 PermissionDenied (file expired or lost permission) and others
                    # If it's a 404 or 403, we should just re-upload.
                    print(f"[DEBUG] Failed to reuse existing file (might be expired or invalid): {e}")
                    # Proceed to upload
                    pass

            # If we couldn't reuse, proceed with upload
            try:
                # Get the appropriate file for upload (OCR text or original PDF)
                file_path, mime_type, needs_cleanup = get_document_for_upload(entry)
                
                if mime_type == "text/plain":
                    print(f"[INFO] Using OCR text for {original_filename}")

                # Upload file
                try:
                    with open(file_path, "rb") as file_handle:
                        uploaded_file = client.files.upload(
                            file=file_handle,
                            config={
                                "mime_type": mime_type,
                                "display_name": original_filename or "document"
                            }
                        )

                    print(f"[DEBUG] Uploaded {original_filename} ({mime_type}) -> uri: {uploaded_file.uri}")
                    
                    # Save the URI (and Name!) to DB for future use
                    # We only added `gemini_file_uri`. We should probably have added `gemini_file_name` too.
                    # But we can extract name from URI or just store name in that column if we want.
                    # For now, let's just use the uploaded file.
                    
                    # Update DB if we have a doc_id
                    if doc_id:
                        try:
                            doc_uuid = uuid.UUID(doc_id)
                            if entry.get("category") == "saved_source":
                                db_source = db.query(ResearchSource).filter(ResearchSource.id == doc_uuid).first()
                                if db_source:
                                    db_source.gemini_file_uri = uploaded_file.uri
                                    db.commit()
                            else:
                                db_doc = db.query(Document).filter(Document.id == doc_uuid).first()
                                if db_doc:
                                    db_doc.gemini_file_uri = uploaded_file.uri
                                    db.commit()
                        except Exception as e:
                            print(f"[WARN] Failed to update DB with Gemini URI: {e}")

                    # Wait for file to be active if it's a PDF
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
    finally:
        db.close()

    return uploaded_files


def _generate_with_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: Optional[str],
    files: List[types.File],
    chat_history: List[Dict[str, str]] = [],
    model: str = "gemini-3-pro-preview"
) -> tuple[str, int]:
    """Call Gemini API and return generated text."""
    
    print(f"[DEBUG] Calling Gemini API:")
    print(f"  Model: {model}")
    print(f"  Files: {len(files)}")
    print(f"  History: {len(chat_history)} messages")
    print(f"  User Prompt: '{user_prompt}'")

    # 1. Prepare effective history and current message
    effective_history = list(chat_history)
    current_msg_text = user_prompt
    
    print(f"[DEBUG] Effective history length: {len(effective_history)}")
    print(f"[DEBUG] Current message text: '{current_msg_text}'")

    # 2. Build Gemini History
    gemini_history = []
    if effective_history:
        # Handle first message with files
        first_entry = effective_history[0]
        first_parts = []
        if files:
            # Convert types.File to types.Part
            for f in files:
                first_parts.append(
                    types.Part(
                        file_data=types.FileData(
                            file_uri=f.uri,
                            mime_type=f.mime_type
                        )
                    )
                )
        
        text = first_entry.get("content", "")
        if text:
            first_parts.append(types.Part.from_text(text=text))
            
        gemini_history.append(types.Content(role="user", parts=first_parts))
        
        # Handle rest
        for entry in effective_history[1:]:
             role = "model" if entry.get("role") == "assistant" else "user"
             text = entry.get("content", "")
             if text:
                 gemini_history.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))

    # 3. Create the chat session
    chat = client.chats.create(
        model=model,
        history=gemini_history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            max_output_tokens=12288,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level="HIGH"
            )
        )
    )

    # 4. Prepare the message to send
    message_parts = []
    
    # If history was empty, files must go here (in the new message)
    # If history was empty, files must go here (in the new message)
    if not effective_history and files:
        for f in files:
            message_parts.append(
                types.Part(
                    file_data=types.FileData(
                        file_uri=f.uri,
                        mime_type=f.mime_type
                    )
                )
            )
        
    if current_msg_text:
        message_parts.append(current_msg_text)

    try:
        response = chat.send_message(message_parts)
        
        total_tokens = 0
        # Log usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            print(f"[DEBUG] Gemini Response - tokens: input={usage.prompt_token_count}, output={usage.candidates_token_count}")
            total_tokens = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

        # Check for Thought Signature (Gemini 3.0)
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'thought_signature') and part.thought_signature:
                    print(f"[DEBUG] Found Thought Signature: {part.thought_signature[:50]}...")
                if hasattr(part, 'thought') and part.thought:
                    print(f"[DEBUG] Found Thought Process: {str(part.thought)[:50]}...")

        return response.text or "", total_tokens

    except Exception as e:
        print(f"[ERROR] Gemini generation failed: {e}")
        # Print more details if available
        if hasattr(e, 'details'):
            print(f"Details: {e.details}")
        raise








def verify_citations_with_llm(
    generated_text: str,
    selected_documents: Dict[str, List[Dict[str, Optional[str]]]],
    gemini_files: Optional[List[types.File]] = None,
) -> Dict[str, List[str]]:
    """
    Verify citations using Gemini 2.5 Flash with a strict, dynamic Pydantic model.
    Checks if cited documents are actually used and if page numbers are correct.
    """
    if not generated_text.strip():
        return {"cited": [], "missing": [], "warnings": ["Generierter Text ist leer."]}

    client = get_gemini_client()
    
    # 1. Prepare expected documents and citation hints
    expected_docs: Dict[str, str] = {} # filename -> citation_hint
    
    # Helper to sanitize filenames for Pydantic field names
    def _sanitize_field_name(name: str) -> str:
        # Replace non-alphanumeric chars with underscore, ensure start with letter
        sanitized = re.sub(r'[^a-zA-Z0-9]', '_', name)
        if not sanitized[0].isalpha():
            sanitized = 'f_' + sanitized
        return sanitized

    # Collect documents with hints
    # Bescheid (Primary)
    for entry in selected_documents.get("bescheid", []):
        filename = entry.get("filename")
        if not filename: continue
        if entry.get("role") == "primary":
            expected_docs[filename] = "Anlage K2"
        else:
            expected_docs[filename] = "Bl. ... der Akte"
            
    # Anhörung
    for entry in selected_documents.get("anhoerung", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Bl. ... der Akte"
            
    # Rechtsprechung
    for entry in selected_documents.get("rechtsprechung", []):
        filename = entry.get("filename")
        if filename:
            expected_docs[filename] = "Urteil / Beschluss"
            
    # Saved Sources
    for entry in selected_documents.get("saved_sources", []):
        title = entry.get("title") or entry.get("id")
        if title:
            expected_docs[title] = "Quelle / Titel"

    if not expected_docs:
        return {"cited": [], "missing": [], "warnings": ["Keine Dokumente zur Verifizierung ausgewählt."]}

    # 2. Create Dynamic Pydantic Model
    field_definitions = {}
    filename_map = {} # field_name -> original_filename
    
    for filename, hint in expected_docs.items():
        field_name = _sanitize_field_name(filename)
        # Handle potential collisions
        counter = 1
        base_field_name = field_name
        while field_name in filename_map:
            field_name = f"{base_field_name}_{counter}"
            counter += 1
            
        filename_map[field_name] = filename
        
        description = f"Wurde das Dokument '{filename}' (z.B. zitiert als '{hint}') im Text verwendet/zitiert?"
        field_definitions[field_name] = (bool, Field(..., description=description))

    # Add warnings field
    field_definitions["warnings"] = (List[str], Field(default_factory=list, description="Liste von Warnungen (z.B. falsche Seitenzahlen, halluzinierte Dokumente)"))
    
    VerificationModel = create_model("VerificationModel", **field_definitions)

    # 3. Construct Prompt
    prompt = f"""Du bist ein strenger juristischer Prüfer. Überprüfe die Zitate im folgenden Text.
    
    TEXT ZUR PRÜFUNG:
    {generated_text}
    
    AUFGABE:
    Prüfe für JEDES der folgenden Dokumente, ob es im Text zitiert oder inhaltlich verwendet wurde.
    Sei streng: Wenn ein Dokument nicht erwähnt wird, setze den Wert auf False.
    Ignoriere Dokumente, die im Text erwähnt werden, aber NICHT in der Liste der erwarteten Dokumente stehen.
    """

    try:
        # If we have files, use them. If not, we can only check text-based references.
        parts = []
        if gemini_files:
            for f in gemini_files:
                # Check if f is already a Part or File object
                if hasattr(f, 'uri'):
                    parts.append(
                        types.Part(
                            file_data=types.FileData(
                                file_uri=f.uri,
                                mime_type=f.mime_type
                            )
                        )
                    )
                else:
                    # Fallback or error logging
                    print(f"[WARN] Unexpected file object type in verification: {type(f)}")

        parts.append(types.Part.from_text(text=prompt))

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-09-2025",
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=VerificationModel,
            ),
        )
        
        # 4. Parse Response
        import json
        if not response.text:
            print(f"[VERIFICATION ERROR] Empty response from Gemini. Candidates: {response.candidates}")
            return {"cited": [], "missing": [], "warnings": ["Verifizierung fehlgeschlagen: Keine Antwort vom Modell."]}
            
        result_dict = json.loads(response.text)
        
        cited = []
        missing = []
        warnings = result_dict.get("warnings", [])
        
        for field_name, original_filename in filename_map.items():
            is_cited = result_dict.get(field_name, False)
            if is_cited:
                cited.append(original_filename)
            else:
                missing.append(original_filename)
                
        return {
            "cited": cited,
            "missing": missing,
            "warnings": warnings,
        }

    except Exception as e:
        print(f"[VERIFICATION ERROR] {e}")
        return {"cited": [], "missing": [], "warnings": [f"Verifizierung fehlgeschlagen: {e}"]}



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
    for category in ("anhoerung", "bescheid", "vorinstanz", "rechtsprechung"):
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

    if body.chat_history:
        # Amelioration: Use raw user prompt, do not wrap in template
        print("[DEBUG] Amelioration detected: Using raw user_prompt")
        system_prompt = "Du bist ein hilfreicher juristischer Assistent." # Simple system prompt or reuse existing?
        # Actually _build_generation_prompts returns system_prompt too. 
        # Let's keep system_prompt but use raw user_prompt.
        sys_p, _ = _build_generation_prompts(
            body, collected, primary_bescheid_label,
            primary_bescheid_description, context_summary
        )
        system_prompt = sys_p
        user_prompt = body.user_prompt
    else:
        system_prompt, user_prompt = _build_generation_prompts(
            body, collected, primary_bescheid_label,
            primary_bescheid_description, context_summary
        )

    # If we are in a conversation (amelioration), the history already contains the context.
    # We should NOT append the original full prompt again, as it would restart the task.
    prompt_for_generation = user_prompt
    # If we are in a conversation (amelioration), we use the user_prompt as the next message.
    # We used to clear it, but now the frontend sends the new instruction as user_prompt.
    prompt_for_generation = user_prompt

    # 4. Route to provider-specific logic
    # 4. Route to provider-specific logic
    try:
        if body.model == "multi-step-expert":
            print(f"[DEBUG] Generation Request Body: {body.model_dump_json(exclude={'selected_documents'})}")
            # Check for amelioration (interactive improvement)
            if body.chat_history:
                print(f"[DEBUG] Multi-Step Expert Amelioration: Switching to Gemini 3")
                client = get_gemini_client()
                files = _upload_documents_to_gemini(client, document_entries)
                print(f"[DEBUG] Uploaded {len(files)} documents to Gemini Files API")

                generated_text, token_count = _generate_with_gemini(
                    client, system_prompt, prompt_for_generation, files,
                    chat_history=body.chat_history,
                    model="gemini-3-pro-preview"
                )
            else:
                print(f"[DEBUG] Starting Multi-Step Expert Workflow")
                
                # Step 1: Draft with Claude
                print(f"[DEBUG] Step 1: Drafting with Claude Opus")
                claude_client = get_anthropic_client()
                document_blocks = _upload_documents_to_claude(claude_client, document_entries)
                
                draft_text, draft_tokens = _generate_with_claude(
                    claude_client, system_prompt, prompt_for_generation, document_blocks,
                    chat_history=body.chat_history
                )
                print(f"[DEBUG] Draft generated ({len(draft_text)} chars)")
                
                # Step 2: Critique with GPT-5.1
                print(f"[DEBUG] Step 2: Critiquing with GPT-5.1")
                openai_client = get_openai_client()
                openai_file_blocks = _upload_documents_to_openai(openai_client, document_entries)
                
                critique_system_prompt = (
                    "Du bist ein kritischer juristischer Prüfer. Analysiere den folgenden Entwurf auf logische Brüche, "
                    "fehlende Argumente aus den Akten und rechtliche Ungenauigkeiten. "
                    "Sei streng und präzise. Liste konkrete Verbesserungsvorschläge auf."
                )
                critique_user_prompt = f"Hier ist der zu prüfende Entwurf:\n\n{draft_text}"
                
                critique_text, critique_tokens = _generate_with_gpt5(
                    openai_client, critique_system_prompt, critique_user_prompt, openai_file_blocks,
                    chat_history=[], # No history for critique
                    reasoning_effort="high",
                    verbosity="low",
                    model="gpt-5.1"
                )
                print(f"[DEBUG] Critique generated ({len(critique_text)} chars)")
                
                # Step 3: Finalize with Gemini 3
                print(f"[DEBUG] Step 3: Finalizing with Gemini 3")
                gemini_client = get_gemini_client()
                gemini_files = _upload_documents_to_gemini(gemini_client, document_entries)
                
                final_system_prompt = (
                    "Du bist ein erfahrener Fachanwalt. Überarbeite den folgenden Entwurf basierend auf der Kritik. "
                    "Behalte die Stärken des Entwurfs bei, aber korrigiere die genannten Schwächen. "
                    "Erstelle die finale, unterschriftsreife Version.\n"
                    "WICHTIG: Verwende KEINE Markdown-Formatierung für Überschriften (wie **Fett** oder ##). "
                    "Nutze stattdessen normale Absätze und Leerzeilen zur Gliederung.\n\n"
                    "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
                    "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
                    "- Nutze die volle verfügbare Länge (bis zu 12.000 Token), um den Sachverhalt umfassend zu würdigen.\n"
                    "- Gehe detailliert auf jeden Widerspruch und jedes Beweismittel ein."
                )
                final_user_prompt = (
                    f"ENTWURF:\n{draft_text}\n\n"
                    f"KRITIK:\n{critique_text}\n\n"
                    "Bitte erstelle nun die finale Version."
                )
                
                generated_text, final_tokens = _generate_with_gemini(
                    gemini_client, final_system_prompt, final_user_prompt, gemini_files,
                    chat_history=[], # No history for finalization, we send context in prompt
                    model="gemini-3-pro-preview"
                )
                
                token_count = draft_tokens + critique_tokens + final_tokens
                # For verification later
                files = gemini_files 
            
        elif body.model.startswith("gpt"):
            # GPT-5.1 path (Responses API)
            print(f"[DEBUG] Using OpenAI GPT-5.1: {body.model}")
            client = get_openai_client()
            file_blocks = _upload_documents_to_openai(client, document_entries)
            print(f"[DEBUG] Uploaded {len(file_blocks)} documents to OpenAI Files API")

            try:
                generated_text, token_count = _generate_with_gpt5(
                    client, system_prompt, prompt_for_generation, file_blocks,
                    chat_history=body.chat_history,
                    reasoning_effort="high",
                    verbosity=body.verbosity,
                    model=body.model
                )
            except Exception as e:
                # Check for rate limit error specifically
                error_str = str(e)
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    # Extract just the limit info if possible, otherwise generic
                    limit_msg = "Limit 30,000 TPM" if "30000" in error_str else "Rate limit exceeded"
                    raise HTTPException(
                        status_code=429,
                        detail=f"⚠️ OpenAI Rate Limit Exceeded ({limit_msg}). Please deselect some documents (e.g. Rechtsprechung) to reduce the request size."
                    )
                raise HTTPException(status_code=500, detail=f"OpenAI API Error: {error_str}")
        elif body.model.startswith("gemini"):
            # Gemini path
            print(f"[DEBUG] Using Google Gemini: {body.model}")
            client = get_gemini_client()
            files = _upload_documents_to_gemini(client, document_entries)
            print(f"[DEBUG] Uploaded {len(files)} documents to Gemini Files API")

            generated_text, token_count = _generate_with_gemini(
                client, system_prompt, prompt_for_generation, files,
                chat_history=body.chat_history,
                model=body.model
            )
        else:
            # Claude path (default)
            print(f"[DEBUG] Using Anthropic Claude: {body.model}")
            client = get_anthropic_client()
            document_blocks = _upload_documents_to_claude(client, document_entries)
            print(f"[DEBUG] Uploaded {len(document_blocks)} documents to Claude Files API")

            generated_text, token_count = _generate_with_claude(
                client, system_prompt, prompt_for_generation, document_blocks,
                chat_history=body.chat_history
            )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generierung fehlgeschlagen: {exc}")

    # 5. Verification (LLM-based)
    # We need to gather files for verification if they weren't already used for generation (Gemini case)
    verification_files = []
    
    # If we used Gemini for generation, we might have 'files' variable populated
    if 'files' in locals() and files:
        verification_files = files
    else:
        # We need to prepare files for verification (upload or reuse URI)
        # This logic should be robust: check DB for URI, else upload
        # For now, let's reuse the _upload_documents_to_gemini logic but modified to check DB
        pass # We will implement this in the helper function or here
        
        # Quick implementation:
        client = get_gemini_client()
        verification_files = _upload_documents_to_gemini(client, document_entries) # This needs to be updated to check DB!

    citations = verify_citations_with_llm(generated_text, collected, gemini_files=verification_files)
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
        token_count=token_count if 'token_count' in locals() else 0,
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
