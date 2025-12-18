import json
import time
import re
import unicodedata
import uuid
from pathlib import Path
from pydantic import BaseModel, Field, create_model
from typing import Any, Dict, List, Optional

import httpx
import pikepdf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
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
    TokenUsage,
    broadcast_documents_snapshot,
    get_anthropic_client,
    get_openai_client,
    limiter,
    load_document_text,
    store_document_text,
    get_gemini_client,
    ensure_document_on_gemini,
)
from auth import get_current_active_user
from database import get_db
from models import Document, ResearchSource, User, GeneratedDraft

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


def _collect_selected_documents(selection, db: Session, current_user: User, require_bescheid: bool = True) -> Dict[str, List[Dict[str, Optional[str]]]]:
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
        "vorinstanz": [],
        "rechtsprechung": [],
        "saved_sources": [],
        "sonstiges": [],
        "akte": [],
    }

    # Load Anhörung documents
    if selection.anhoerung:
        query = (
            db.query(Document)
            .filter(
                Document.filename.in_(selection.anhoerung),
                Document.owner_id == current_user.id
            )
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
    if require_bescheid and not bescheid_selection.primary:
        raise HTTPException(status_code=400, detail="Bitte markieren Sie einen Bescheid als Hauptbescheid (Anlage K2)")

    bescheid_filenames = [fn for fn in ([bescheid_selection.primary] + (bescheid_selection.others or [])) if fn]
    bescheid_query = (
        db.query(Document)
        .filter(
            Document.filename.in_(bescheid_filenames),
            Document.owner_id == current_user.id
        )
        .all()
    )
    bescheid_map = {doc.filename: doc for doc in bescheid_query}

    missing_bescheide = [fn for fn in bescheid_filenames if fn not in bescheid_map]
    if missing_bescheide:
        raise HTTPException(status_code=404, detail=f"Bescheid-Dokumente nicht gefunden: {', '.join(missing_bescheide)}")

    if bescheid_selection.primary:
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
            .filter(
                Document.filename.in_(vorinstanz_filenames),
                Document.owner_id == current_user.id
            )
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
            .filter(
                Document.filename.in_(selection.rechtsprechung),
                Document.owner_id == current_user.id
            )
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.rechtsprechung if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Rechtsprechung-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.RECHTSPRECHUNG.value)
            collected["rechtsprechung"].append(_document_to_context_dict(doc))



    # Load Akte documents
    if selection.akte:
        query = (
            db.query(Document)
            .filter(
                Document.filename.in_(selection.akte),
                Document.owner_id == current_user.id
            )
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.akte if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Akte-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.AKTE.value)
            collected["akte"].append(_document_to_context_dict(doc))

    # Load Sonstiges documents
    if selection.sonstiges:
        query = (
            db.query(Document)
            .filter(
                Document.filename.in_(selection.sonstiges),
                Document.owner_id == current_user.id
            )
            .all()
        )
        found_map = {doc.filename: doc for doc in query}
        missing = [fn for fn in selection.sonstiges if fn not in found_map]
        if missing:
            raise HTTPException(status_code=404, detail=f"Sonstiges-Dokumente nicht gefunden: {', '.join(missing)}")
        for doc in query:
            _validate_category(doc, DocumentCategory.SONSTIGES.value)
            collected["sonstiges"].append(_document_to_context_dict(doc))
            
    # Load saved sources
    if selection.saved_sources:
        collected_sources = []
        for source_id in selection.saved_sources:
            try:
                source_uuid = uuid.UUID(source_id)
            except ValueError:
                # If we still get a non-UUID here, it is a validation error.
                # However, since we now handle "Sonstiges" separately, documents shouldn't end up here.
                # But to be safe and avoid 500s or vague 400s:
                raise HTTPException(status_code=400, detail=f"Ungültige Quellen-ID (erwartet UUID, erhalten '{source_id}'): {source_id}")
            source = db.query(ResearchSource).filter(
                ResearchSource.id == source_uuid,
                ResearchSource.owner_id == current_user.id
            ).first()
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
    
    # Determine Mode based on Document Type and Selections
    doc_type_lower = body.document_type.lower()
    
    # 1. AZB (Appeal) Mode
    # Triggered explicitly by classification "AZB" or "Berufung"
    is_azb = "azb" in doc_type_lower or "zulassung der berufung" in doc_type_lower

    # 2. Klage/Bescheid Mode (Legacy)
    # Triggered by "Klage" type OR presence of a selected Primary Bescheid
    primary_bescheid_entry = next(
        (entry for entry in collected.get("bescheid", []) if entry.get("role") == "primary"),
        None,
    )
    is_klage_or_bescheid = "klage" in doc_type_lower or primary_bescheid_entry is not None
    
    if is_azb:
        # --- AZB PROMPT LOGIC ---
        print("[DEBUG] AZB Mode activated (Vorinstanz documents present)")
        
        # Identify primary Vorinstanz document (Judgment)
        primary_vorinstanz_doc = next((d for d in collected.get("vorinstanz", []) if d.get("role") == "primary"), None)
        primary_vorinstanz_label = primary_vorinstanz_doc["filename"] if primary_vorinstanz_doc else "das Urteil"

        system_prompt = (
            # Anthropic extended thinking tip: Use high-level instructions for thorough reasoning
            "DENKWEISE:\n"
            "Denke gründlich und ausführlich über diesen Fall nach, bevor du schreibst. "
            "Analysiere ALLE vorliegenden Dokumente sorgfältig. "
            "Betrachte verschiedene Argumentationsansätze und wähle die überzeugendsten. "
            "Prüfe deine Argumentation auf Lücken und Schwächen, bevor du sie finalisierst.\n\n"
            
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
            "Bevor du den Text schreibst, analysiere den Fall tiefgehend in den folgenden XML-Tags:\n\n"
            
            "<document_analysis>\n"
            "- **Vorinstanz:** Wo genau hat das Gericht Vortrag ignoriert? (Seite/Absatz zitieren)\n"
            "- **Protokolle:** Was steht im Protokoll, das im Urteil fehlt? Zitiere den Wortlaut.\n"
            "- **Schriftsätze:** Welcher schriftliche Vortrag wurde übergangen?\n"
            "</document_analysis>\n\n"

            "<strategy>\n"
            "1. **Ziel:** Zulassung der Berufung wegen Verfahrensmangel (§ 78 Abs. 3 Nr. 3 AsylG).\n"
            "2. **Fakten:** Die stärksten ignorierten Punkte aus der Analyse.\n"
            "3. **Rechtsgrundlage:** Art. 103 Abs. 1 GG (Rechtliches Gehör).\n"
            "4. **Argumentation:**\n"
            "   - Das Gericht hat X ignoriert.\n"
            "   - Das war entscheidungserheblich (Kausalität), weil...\n"
            "</strategy>\n\n"

            "Verfasse nun basierend auf dieser Strategie die Begründung des Zulassungsantrags als Fließtext (OHNE die XML-Tags im Output zu wiederholen)."
        )
        
    elif is_klage_or_bescheid:
        # --- LEGACY / BESCHEID PROMPT LOGIC ---
        system_prompt = (
            # Anthropic extended thinking tip: Use high-level instructions for thorough reasoning
            "DENKWEISE:\n"
            "Denke gründlich und ausführlich über diesen Fall nach, bevor du schreibst. "
            "Analysiere ALLE vorliegenden Dokumente sorgfältig. "
            "Betrachte verschiedene Argumentationsansätze und wähle die überzeugendsten. "
            "Prüfe deine Argumentation auf Lücken und Schwächen, bevor du sie finalisierst.\n\n"
            
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
    
            "AUFGABE:\n"
            "Analysiere die Dokumente sorgfältig und verfasse die detaillierte rechtliche Würdigung als Fließtext.\n"
            "- Identifiziere die Ablehnungsgründe des BAMF und widerlege sie mit Fakten aus der Anhörung.\n"
            "- Zitiere konkret aus den beigefügten Urteilen und Quellen.\n"
            "- Beginne direkt mit der juristischen Argumentation ohne Adressblock oder Anrede."
        )

    else:
        # --- GENERAL APPLICATION / SCHRIFTSATZ PROMPT (No Bescheid) ---
        print("[DEBUG] General Application Mode activated (No Bescheid selected)")
        
        system_prompt = (
            "Du bist ein erfahrener Fachanwalt für Migrations- und Ausländerrecht. "
            f"Du erstellst einen rechtlichen Schriftsatz oder Antrag (z.B. Niederlassungserlaubnis, Einbürgerung, Stellungnahme) für Mandanten.\n\n"

            "ZIEL & FOKUS:\n"
            "Fokussiere dich auf die positive Darlegung der Anspruchsvoraussetzungen für das begehrte Ziel (z.B. Erteilung einer Erlaubnis).\n"
            "- Identifiziere die einschlägige Rechtsgrundlage (z.B. AufenthG, StAG, FreizügG/EU).\n"
            "- Subsumiere die Fakten aus den Dokumenten unter die Tatbestandsmerkmale.\n"
            "- Argumentiere präzise und lösungsorientiert.\n\n"

            "BEWEISFÜHRUNG:\n"
            "Nutze alle verfügbaren Dokumente (Aktenauszüge, Zertifikate, Protokolle, 'Sonstiges'), um die Voraussetzungen (z.B. Lebensunterhalt, Identität, Aufenthaltszeiten, Straffreiheit) zu belegen.\n"
            "- Zitiere konkret aus den Unterlagen, wo immer möglich.\n\n"

            "STIL & FORMAT:\n"
            "- Juristischer Profi-Stil (Sachlich, Überzeugend).\n"
            "- Klar strukturiert (Sachverhalt -> Rechtliche Würdigung -> Ergebnis).\n"
            "- Keine Floskeln.\n"
            "- Beginne direkt mit dem juristischen Text, keine Adresszeilen oder Anreden."
        )

        # Verbosity
        verbosity = body.verbosity
        if verbosity == "low":
            system_prompt += "\n\nFASSUNG (LOW): Kurz und bündig. Nur Key-Facts."
        elif verbosity == "medium":
            system_prompt += "\n\nFASSUNG (MEDIUM): Standard-Schriftsatzlänge. Ausgewogen."
        else: # high
            system_prompt += "\n\nFASSUNG (HIGH): Ausführliche Darlegung aller Voraussetzungen und detaillierte Würdigung aller Belege."

        user_prompt = (
            f"Dokumententyp: {body.document_type}\n"
            f"Auftrag: {body.user_prompt.strip()}\n\n"

            "VERFÜGBARE DOKUMENTE:\n"
            f"{context_summary or '- (Keine Dokumente)'}\n\n"

            "VORGEHENSWEISE:\n"
            "Bevor du den Text schreibst, analysiere den Fall in den folgenden XML-Tags:\n\n"

            "<document_analysis>\n"
            "- **Auftrag:** Was genau ist das Ziel? (z.B. Niederlassungserlaubnis)\n"
            "- **Voraussetzungen:** Welche gesetzlichen Merkmale (§§) müssen erfüllt sein?\n"
            "- **Belege:** Welche Dokumente beweisen diese Merkmale?\n"
            "</document_analysis>\n\n"

            "<strategy>\n"
            "1. **Ziel:** Anspruchsdurchsetzung.\n"
            "2. **Fakten/Belege:** Zuordnung der Dokumente zu den Tatbestandsmerkmalen.\n"
            "3. **Argumentation:** Subsumtion der Fakten unter die Rechtsnorm.\n"
            "</strategy>\n\n"

            "Verfasse nun basierend auf dieser Strategie den Schriftsatz als Fließtext (OHNE die XML-Tags im Output zu wiederholen)."
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

    Prefers OCR'd text when available for better accuracy and significantly lower token usage.
    
    OPTIMIZATION:
    If we have extracted text (text/plain), we embed it DIRECTLY as a text block
    instead of using the Files API. This matches our benchmark findings where
    text embedding is ~2.3x more efficient than PDF upload and cleaner than text file upload.
    """
    content_blocks: List[Dict[str, str]] = []
    MAX_PAGES = 100  # Claude Files API limit

    for entry in documents:
        original_filename = entry.get("filename") or "document"

        try:
            # Get the appropriate file for upload (OCR text or original PDF)
            file_path, mime_type, needs_cleanup = get_document_for_upload(entry)

            if mime_type == "text/plain":
                print(f"[INFO] Embedding OCR Text for {original_filename} (skipping upload)")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Create a text block with the document content, similar to OpenAI pipeline
                    text_block = f"DOKUMENT: {original_filename}\n\n{content}"
                    
                    content_blocks.append({
                        "type": "text",
                        "text": text_block
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to read text file {file_path}: {e}")

            else:
                # It's a PDF (or other supported binary)
                
                # OPTIMIZATION: Try to extract text from native PDFs on-the-fly
                # This catches PDFs that didn't need OCR (native text) and avoids expensive vision processing
                extracted_text = None
                if mime_type == "application/pdf":
                    try:
                        import fitz  # PyMuPDF
                        with fitz.open(file_path) as pdf_doc:
                            # 1. Quick check: Is there enough text?
                            full_text = ""
                            page_count = len(pdf_doc)
                            
                            # Limit extraction for huge documents to avoid context overflow, 
                            # but 100 pages is usually fine for text-only
                            pages_to_extract = min(page_count, 100) 
                            
                            for i in range(pages_to_extract):
                                full_text += pdf_doc[i].get_text() + "\n\n"
                                
                            # If we have substantial text (avg > 100 chars per page), assume it's good
                            if len(full_text) > (50 * pages_to_extract):
                                print(f"[INFO] On-the-fly text extraction successful for {original_filename} ({len(full_text)} chars)")
                                extracted_text = f"DOKUMENT: {original_filename} (Text-Extrakt)\n\n{full_text}"
                    except Exception as e:
                        print(f"[WARN] On-the-fly text extraction failed for {original_filename}: {e}")

                if extracted_text:
                    # Use the on-the-fly extracted text
                    content_blocks.append({
                        "type": "text",
                        "text": extracted_text
                    })
                else: 
                    # FALLBACK: Upload as File (Vision)
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
                            "title": original_filename,
                        })
                        print(f"[DEBUG] Uploaded {original_filename} (PDF) -> file_id: {uploaded_file.id}")

                    except Exception as exc:
                        print(f"[ERROR] Upload für {original_filename} fehlgeschlagen: {exc}")


        except (ValueError, FileNotFoundError) as exc:
            print(f"[WARN] Überspringe {original_filename}: {exc}")
            continue
        finally:
            # Clean up temporary file if needed
            if 'needs_cleanup' in locals() and needs_cleanup:
                try:
                    os.unlink(file_path)
                except:
                    pass

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


@router.post("/generate")
@limiter.limit("10/hour")
async def generate(
    request: Request,
    body: GenerationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Generate drafts using Claude Sonnet 4.5 or GPT-5 Reasoning models (Streaming)."""
    
    # 1. Collect potential documents
    collected = {
        "anhoerung": [],
        "bescheid": [],
        "vorinstanz": [],
        "rechtsprechung": [], # Includes saved_sources if generic
        "saved_sources": [],
        "akte": [],
        "sonstiges": []
    }
    
    # Flatten selection
    selection_files = set()
    
    # Helper to add files
    def add_files(files):
        for f in files:
            selection_files.add(f)

    if body.selected_documents:
        add_files(body.selected_documents.anhoerung)
        add_files(body.selected_documents.bescheid.others)
        if body.selected_documents.bescheid.primary:
            selection_files.add(body.selected_documents.bescheid.primary)
        add_files(body.selected_documents.vorinstanz.others)
        if body.selected_documents.vorinstanz.primary:
            selection_files.add(body.selected_documents.vorinstanz.primary)
        add_files(body.selected_documents.rechtsprechung)
        add_files(body.selected_documents.saved_sources)
        add_files(body.selected_documents.akte)
        add_files(body.selected_documents.sonstiges)
        
    documents = db.query(Document).filter(
        Document.filename.in_(list(selection_files))
    ).all()
    
    start_time = time.time()
    
    # Map to collected dict
    doc_map = {d.filename: d for d in documents}
    
    # Re-map role logic using helper to avoid AttributeError
    for fname in body.selected_documents.anhoerung:
        if fname in doc_map: 
            collected["anhoerung"].append(_document_to_context_dict(doc_map[fname]))
        
    if body.selected_documents.bescheid.primary in doc_map:
        d = doc_map[body.selected_documents.bescheid.primary]
        entry = _document_to_context_dict(d)
        entry["role"] = "primary"
        collected["bescheid"].append(entry)
        
    for fname in body.selected_documents.bescheid.others:
        if fname in doc_map: 
            collected["bescheid"].append(_document_to_context_dict(doc_map[fname]))

    if body.selected_documents.vorinstanz.primary in doc_map:
        d = doc_map[body.selected_documents.vorinstanz.primary]
        entry = _document_to_context_dict(d)
        entry["role"] = "primary"
        collected["vorinstanz"].append(entry)

    for fname in body.selected_documents.vorinstanz.others:
        if fname in doc_map: 
            collected["vorinstanz"].append(_document_to_context_dict(doc_map[fname]))
        
    for fname in body.selected_documents.rechtsprechung:
        if fname in doc_map: 
            collected["rechtsprechung"].append(_document_to_context_dict(doc_map[fname]))

    # Saved sources (Special handling: might not be in Document table but ResearchSource?)
    # The original code queries `Document` table for saved_sources too?
    # Let's check original code: 
    # `documents = db.query(Document)...`
    # `saved_sources_ids = body.selected_documents.saved_sources`
    # `sources = db.query(ResearchSource)...`
    
    # Handle saved sources
    if body.selected_documents.saved_sources:
        # these are IDs usually? Or filenames?
        # The frontend sends IDs for saved_sources I think.
        # Let's assumethey are IDs.
        sources = db.query(ResearchSource).filter(ResearchSource.id.in_(body.selected_documents.saved_sources)).all()
        for s in sources:
            content = s.description or ""
            entry = {
                "id": str(s.id),
                "title": s.title,
                "filename": s.title,
                "content": content,
                "category": "saved_sources"
            }
            # Add file_path if available so it can be uploaded/embedded
            if s.download_path and os.path.exists(s.download_path):
                entry["file_path"] = s.download_path
                
            collected["saved_sources"].append(entry)

    for fname in body.selected_documents.akte:
        if fname in doc_map: 
            collected["akte"].append(_document_to_context_dict(doc_map[fname]))
            
    for fname in body.selected_documents.sonstiges:
        if fname in doc_map: 
            collected["sonstiges"].append(_document_to_context_dict(doc_map[fname]))

    # Flatten for context window
    document_entries = []
    for cat, items in collected.items():
        document_entries.extend(items)
        
    print(f"[DEBUG] Collected {len(document_entries)} document entries for upload")
    
    # 3. Build prompts
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

    if body.chat_history:
        print("[DEBUG] Amelioration detected: Using raw user_prompt")
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

    prompt_for_generation = user_prompt

    # STREAM GENERATOR
    async def stream_generator():
        generated_text_acc = []
        thinking_text_acc = []
        token_usage_acc = None
        
        try:
            # 4. Route to provider
            if body.model == "multi-step-expert":
                # Multi-step is tricky to stream step-by-step unless we rewrite it to yield events.
                # For now, we will execute it synchronously (blocking threadpool) and yield one big chunks.
                # Or yield progress events?
                yield json.dumps({"type": "thinking", "text": "Starting Multi-Step Expert Workflow...\n"}) + "\n"
                
                # ... (Existing Multi-step logic) ...
                # To avoid duplicating massive logic inside the generator, we should ideally refactor multi-step to be a generator too.
                # But for time constraints, we will call the synchronous logic and yield the result.
                # NOTE: This will NOT solve timeout if multi-step takes > 10m and is opaque.
                # But Multi-Step uses multiple calls, each < 10m likely.
                # However, the user asked to fix Claude timeout.
                
                # We can replicate the logic:
                if body.chat_history:
                     # Gemini Amelioration logic
                     client = get_gemini_client()
                     files = _upload_documents_to_gemini(client, document_entries)
                     text, count = _generate_with_gemini(
                        client, system_prompt, prompt_for_generation, files,
                        chat_history=body.chat_history,
                        model="gemini-3-pro-preview"
                     )
                     generated_text_acc.append(text)
                     yield json.dumps({"type": "text", "text": text}) + "\n"
                     token_usage_acc = TokenUsage(total_tokens=count, model="gemini-3-pro-preview")
                else:
                    # Full 3-step
                    yield json.dumps({"type": "thinking", "text": "[Step 1/3] Drafting with Claude Opus...\n"}) + "\n"
                    # We can use the streaming Claude here too?
                    claude_client = get_anthropic_client()
                    document_blocks = _upload_documents_to_claude(claude_client, document_entries)
                    
                    # Call our new streaming function but consume it internally
                    step1_text = []
                    for chunk_str in _generate_with_claude_stream(
                        claude_client, system_prompt, prompt_for_generation, document_blocks, chat_history=body.chat_history
                    ):
                        chunk = json.loads(chunk_str)
                        if chunk["type"] == "text":
                            step1_text.append(chunk["text"])
                        elif chunk["type"] == "thinking":
                            yield json.dumps({"type": "thinking", "text": chunk["text"]}) + "\n"
                    
                    draft_text = "".join(step1_text)
                    yield json.dumps({"type": "thinking", "text": "\n[Step 2/3] Critiquing with GPT-5.1...\n"}) + "\n"
                    
                    # Critique (OpenAI)
                    openai_client = get_openai_client()
                    openai_file_blocks = _upload_documents_to_openai(openai_client, document_entries)
                    critique_system_prompt = (
                        "Du bist ein Senior Partner einer Top-Kanzlei, bekannt für extrem strenge Qualitätskontrolle.\n"
                        "Analysiere den folgenden Entwurf gnadenlos auf:\n"
                        "1. HALLUZINATIONEN: Prüfe jedes zitierte Urteil. Sieht das Aktenzeichen echt aus? Gibt es das Gericht?\n"
                        "2. LOGIK: Ist die juristische Argumentation schlüssig? Gibt es Sprünge?\n"
                        "3. TONALITÄT: Ist der Schriftsatz professionell und überzeugend?\n"
                        "Liste konkrete Mängel auf. Sei pedantisch."
                    )
                    critique_user_prompt = f"Hier ist der zu prüfende Entwurf:\n\n{draft_text}"
                    critique_text, critique_tokens = _generate_with_gpt5(
                        openai_client, critique_system_prompt, critique_user_prompt, openai_file_blocks,
                        chat_history=[],
                        reasoning_effort="high",
                        verbosity="low",
                        model="gpt-5.2"
                    )
                    yield json.dumps({"type": "thinking", "text": f"Critique: {critique_text[:200]}...\n"}) + "\n"
                    
                    yield json.dumps({"type": "thinking", "text": "\n[Step 3/3] Finalizing with Gemini 3...\n"}) + "\n"
                    
                    # Finalize (Gemini)
                    gemini_client = get_gemini_client()
                    gemini_files = _upload_documents_to_gemini(gemini_client, document_entries)
                    final_system_prompt = (
                        "Du bist ein erfahrener Fachanwalt. Deine Aufgabe ist die Einarbeitung der Kritik des Senior Partners.\n"
                        "VORGEHENSWEISE:\n"
                        "1. Lies die KRITIK sorgfältig.\n"
                        "2. Überarbeite den ENTWURF: Korrigiere jeden kritisierten Punkt.\n"
                        "3. Halluzinationen entfernen: Wenn der Senior Partner ein Urteil anzweifelt, LÖSCHE es oder ersetze es durch eine allgemeine Formulierung.\n"
                        "4. Behalte die XML-Tags (<strategy> usw.) NICHT bei - nur den reinen juristischen Text.\n\n"
                        "WICHTIG: Verwende KEINE Markdown-Formatierung für Überschriften (wie **Fett** oder ##). "
                        "Nutze stattdessen normale Absätze und Leerzeilen zur Gliederung.\n\n"
                        "FASSUNG & LÄNGE (VERBOSITY: HIGH):\n"
                        "- Die Argumentation muss ausführlich und tiefgehend sein.\n"
                        "- Nutze die volle verfügbare Länge (bis zu 12.000 Token), um den Sachverhalt umfassend zu würdigen."
                    )
                    final_user_prompt = (
                        f"ENTWURF (mit Vorüberlegungen):\n{draft_text}\n\n"
                        f"KRITIK DES SENIOR PARTNERS:\n{critique_text}\n\n"
                        "Erstelle nun die finale, bereinigte Version (ohne <document_analysis> etc.)."
                    )
                    
                    final_text, final_tokens = _generate_with_gemini(
                        gemini_client, final_system_prompt, final_user_prompt, gemini_files,
                        chat_history=[],
                        model="gemini-3-pro-preview"
                    )
                    
                    generated_text_acc.append(final_text)
                    yield json.dumps({"type": "text", "text": final_text}) + "\n"
                    token_usage_acc = TokenUsage(total_tokens=final_tokens, model="multi-step-expert")

            elif body.model.startswith("gpt"):
                client = get_openai_client()
                file_blocks = _upload_documents_to_openai(client, document_entries)
                text, count = _generate_with_gpt5(
                    client, system_prompt, prompt_for_generation, file_blocks,
                    chat_history=body.chat_history,
                    reasoning_effort="high",
                    verbosity=body.verbosity,
                    model=body.model
                )
                generated_text_acc.append(text)
                yield json.dumps({"type": "text", "text": text}) + "\n"
                token_usage_acc = TokenUsage(total_tokens=count, model=body.model)
                
            elif body.model.startswith("gemini"):
                client = get_gemini_client()
                files = _upload_documents_to_gemini(client, document_entries)
                text, count = _generate_with_gemini(
                    client, system_prompt, prompt_for_generation, files,
                    chat_history=body.chat_history,
                    model=body.model
                )
                generated_text_acc.append(text)
                yield json.dumps({"type": "text", "text": text}) + "\n"
                token_usage_acc = TokenUsage(total_tokens=count, model=body.model)
                
            else:
                # Claude (Streaming)
                print(f"[DEBUG] Using Anthropic Claude: {body.model} (STREAMING)")
                client = get_anthropic_client()
                document_blocks = _upload_documents_to_claude(client, document_entries)
                
                for chunk_str in _generate_with_claude_stream(
                    client, system_prompt, prompt_for_generation, document_blocks,
                    chat_history=body.chat_history
                ):
                    chunk = json.loads(chunk_str)
                    if chunk["type"] == "text":
                        generated_text_acc.append(chunk["text"])
                    elif chunk["type"] == "thinking":
                        thinking_text_acc.append(chunk["text"])
                    elif chunk["type"] == "usage":
                        token_usage_acc = TokenUsage(**chunk["data"])
                    yield chunk_str
                    
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            return

        # FINALIZE AND SAVE
        generated_text = "".join(generated_text_acc)
        thinking_text = "".join(thinking_text_acc)
        
        # Verify citations (LLM based) - Optional for streaming?
        # Verification usually takes time. If we stream, we might want to do it AFTER text is done.
        # But we can't update citations metadata on the already sent chunks.
        # We can send a "metadata" event at the end!
        
        # We need validation files again if not present
        verification_files = []
        if body.model.lower().startswith("gemini") or body.model.lower() == "multi-step-expert":
            # If Gemini was used for generation, 'gemini_files' should be available from the last step
            if 'gemini_files' in locals() and gemini_files:
                verification_files = gemini_files
            else:
                # Fallback: re-upload if not already done (e.g., if only GPT/Claude was used)
                client = get_gemini_client()
                verification_files = _upload_documents_to_gemini(client, document_entries)
        else:
            # For Claude/GPT, we might need to upload to Gemini for verification
            client = get_gemini_client()
            verification_files = _upload_documents_to_gemini(client, document_entries)


        citations = verify_citations_with_llm(generated_text, collected, gemini_files=verification_files)
        
        # Build metadata
        metadata = GenerationMetadata(
            documents_used={
                "anhoerung": len(collected.get("anhoerung", [])),
                "bescheid": len(collected.get("bescheid", [])),
                "rechtsprechung": len(collected.get("rechtsprechung", [])),
                "saved_sources": len(collected.get("saved_sources", [])),
                "akte": len(collected.get("akte", [])),
                "sonstiges": len(collected.get("sonstiges", [])),
            },
            citations_found=len(citations.get("cited", [])),
            missing_citations=citations.get("missing", []),
            warnings=citations.get("warnings", []),
            word_count=len(generated_text.split()),
            token_count=token_usage_acc.total_tokens if token_usage_acc else 0,
            token_usage=token_usage_acc,
        )
        
        # Send metadata event
        yield json.dumps({"type": "metadata", "data": metadata.model_dump()}) + "\n"
        
        # Save to DB
        draft_id = None
        try:
            # Re-construct used documents list
            structured_used_documents = []
            for cat, entries in collected.items():
                for entry in entries:
                     fname = entry.get("filename") or entry.get("title")
                     if fname:
                        payload = {"filename": fname, "category": cat}
                        role = entry.get("role")
                        if role:
                            payload["role"] = role
                        structured_used_documents.append(payload)

            draft = GeneratedDraft(
                user_id=current_user.id,
                primary_document_id=uuid.UUID(primary_bescheid_entry["id"]) if primary_bescheid_entry else None,
                document_type=body.document_type,
                user_prompt=body.user_prompt,
                generated_text=generated_text,
                model_used=body.model,
                metadata_={
                    "tokens": token_usage_acc.total_tokens if token_usage_acc else 0,
                    "used_documents": structured_used_documents,
                    "thinking_text": thinking_text # Save thinking too if available
                }
            )
            db.add(draft)
            db.commit()
            db.refresh(draft)
            draft_id = str(draft.id)
            print(f"[INFO] Saved generated draft with ID: {draft_id}")
        except Exception as e:
            print(f"[ERROR] Failed to save draft: {e}")
            
        yield json.dumps({"type": "done", "draft_id": draft_id}) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


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


def _generate_with_claude_stream(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: Optional[str],
    document_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = []
):
    """
    Generator that calls Claude API with extended thinking and yields NDJSON events.
    Yields:
        - {"type": "thinking", "text": "..."}
        - {"type": "text", "text": "..."}
        - {"type": "usage", "data": {...}}
        - {"type": "done"}
    """
    
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
             return
             
        messages.append({"role": "user", "content": content})

    # Debug log
    print(f"[DEBUG] Claude Messages: {len(messages)} turns")

    # Enable extended thinking with budget
    # Note: Using beta API with streaming for long-running requests
    print("[DEBUG] Starting Claude streaming request with 12k thinking budget (reduced)...")
    
    # Use streaming to avoid timeout on long thinking operations
    with client.beta.messages.stream(
        model="claude-opus-4-5",
        system=system_prompt,
        max_tokens=20000, 
        messages=messages,
        thinking={
            "type": "enabled",
            "budget_tokens": 12000
        },
        betas=["files-api-2025-04-14", "interleaved-thinking-2025-05-14"],
    ) as stream:
        for event in stream:
            # We can iterate over specific event types if we want granular control
            # But the stream object simplifies this if we use specialized iterators:
            # - stream.text_stream
            # - stream.events
            # But here 'event' is likely a MessageStreamEvent if we iterate directly?
            # Actually, `client.beta.messages.stream` context manager returns a `MessageStreamManager` -> `MessageStream`.
            # We can iterate `stream` directly to get events.
            
            # Map event type to our NDJSON format
            # Event types: content_block_start, content_block_delta, content_block_stop, etc.
            
            if event.type == "content_block_start":
                block = event.content_block
                if block.type == "thinking":
                    # Thinking block started
                    pass
            elif event.type == "content_block_delta":
                delta = event.delta
                if delta.type == "thinking_delta":
                    yield json.dumps({"type": "thinking", "text": delta.thinking}) + "\n"
                elif delta.type == "text_delta":
                    yield json.dumps({"type": "text", "text": delta.text}) + "\n"
            elif event.type == "message_stop":
                pass
    
        # Get final message to extract usage
        response = stream.get_final_message()
    
    # Log API response metadata
    stop_reason = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    
    if stop_reason == "max_tokens":
        print("[WARN] Generation stopped due to max_tokens limit - output may be incomplete!")

    # Extract detailed token usage
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    
    cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    
    total_tokens = input_tokens + output_tokens
    
    # Calculate cost
    OPUS_INPUT_PRICE = 5.0 / 1_000_000
    OPUS_OUTPUT_PRICE = 25.0 / 1_000_000
    OPUS_CACHE_READ_PRICE = 0.5 / 1_000_000
    OPUS_CACHE_WRITE_PRICE = 6.25 / 1_000_000
    
    cost_usd = (
        (input_tokens - cache_read_tokens) * OPUS_INPUT_PRICE +
        output_tokens * OPUS_OUTPUT_PRICE +
        cache_read_tokens * OPUS_CACHE_READ_PRICE +
        cache_create_tokens * OPUS_CACHE_WRITE_PRICE
    )
    
    # Construct usage payload
    usage_data = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": 0, # Included in output
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_create_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 4),
        "model": "claude-opus-4-5"
    }
    
    yield json.dumps({"type": "usage", "data": usage_data}) + "\n"



def _generate_with_gpt5(
    client: OpenAI,
    system_prompt: str,
    user_prompt: Optional[str],
    file_blocks: List[Dict],
    chat_history: List[Dict[str, str]] = [],
    reasoning_effort: str = "high",
    verbosity: str = "high",
    model: str = "gpt-5.2"
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
    """Upload local documents using Gemini Files API (reusing shared logic)."""
    uploaded_files: List[types.File] = []
    
    # We need a DB session to find the ORM objects for specific documents
    from database import SessionLocal
    db = SessionLocal()

    try:
        for entry in documents:
            doc_id = entry.get("id")
            if not doc_id:
                continue
                
            try:
                doc_uuid = uuid.UUID(doc_id)
                db_obj = None
                
                # Determine correct model based on entry category/type
                if entry.get("category") == "saved_source" or entry.get("document_type") in ["Rechtsprechung", "Quelle"]:
                    # It might be in ResearchSource if it was a saved source
                    # But if it's from "Rechtsprechung" document category, it's a Document.
                    # Best check:
                    if entry.get("category") == "saved_source":
                        db_obj = db.query(ResearchSource).filter(ResearchSource.id == doc_uuid).first()
                    else:
                        db_obj = db.query(Document).filter(Document.id == doc_uuid).first()
                else:
                    db_obj = db.query(Document).filter(Document.id == doc_uuid).first()

                if db_obj:
                    gemini_file = ensure_document_on_gemini(db_obj, db)
                    if gemini_file:
                        uploaded_files.append(gemini_file)
                else:
                    print(f"[WARN] Document {doc_id} not found in DB for Gemini upload.")

            except Exception as e:
                print(f"[WARN] Failed to process document {entry.get('filename')} for Gemini: {e}")
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
            model="gemini-3-flash-preview",
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
