# Klagebegründung/Schriftsatz Generation Feature

## Overview

This feature allows users to generate legal documents (Klagebegründung or Schriftsatz) by selecting relevant source documents and providing specific instructions. The system uses Claude AI to create professionally formatted legal texts in German that can be sent to j-lawyer for template population.

## User Flow

1. **Document Selection**: User selects documents from each category using checkboxes:
   - **Anhörung** files (multiple selection allowed)
   - **Bescheid** files (multiple selection, user marks ONE as primary for Anlage K2)
   - **Rechtsprechung** files (multiple selection)
   - **Gespeicherte Quellen** (multiple selection from saved sources)

2. **Prompt Input**: User enters specific instructions describing what arguments to make, key points to address, etc.

3. **Generation**: User clicks "Klagebegründung generieren" button

4. **Preview & Edit**: System displays generated text in modal with edit capability

5. **Send to j-lawyer**: User clicks "An j-lawyer senden" to push to j-lawyer API

## Backend Architecture

### API Endpoints

1. `POST /generate` – erstellt den Entwurf samt Zitier-Metadaten
2. `GET /jlawyer/templates` – listet verfügbare j-lawyer ODT-Templates (Standardordner konfigurierbar)
3. `POST /send-to-jlawyer` – überträgt den Entwurf in die j-lawyer Vorlage

**Rate Limit:** 10 requests per hour

**Request Structure:**
```json
{
  "document_type": "Klagebegründung" | "Schriftsatz",
  "user_prompt": "Beantrage Feststellung des Abschiebungsverbots nach § 60 Abs. 5 AufenthG...",
  "selected_documents": {
    "anhoerung": ["filename1.pdf", "filename2.pdf"],
    "bescheid": {
      "primary": "bescheid1.pdf",
      "others": ["bescheid2.pdf", "bescheid3.pdf"]
    },
    "rechtsprechung": ["urteil1.pdf", "urteil2.pdf"],
    "saved_sources": ["uuid1", "uuid2", "uuid3"]
  }
}
```

**Response Structure:**
```json
{
  "success": true,
  "document_type": "Klagebegründung",
  "user_prompt": "Beantrage Feststellung des Abschiebungsverbots...",
  "generated_text": "I. Sachverhalt\n\n...",
  "used_documents": [
    {"filename": "bescheid1.pdf", "category": "bescheid", "role": "primary"},
    {"filename": "urteil1.pdf", "category": "rechtsprechung"}
  ],
  "metadata": {
    "documents_used": {
      "anhoerung": 2,
      "bescheid": 3,
      "rechtsprechung": 2,
      "saved_sources": 3
    },
    "citations_found": 12,
    "missing_citations": ["Anhörung: anhoerung1.pdf"],
    "warnings": ["Bescheid: bescheid1.pdf – Referenz 'Anlage K2' nicht gefunden"],
    "word_count": 2450
  }
}
```

**Request (`POST /send-to-jlawyer`):**
```json
{
  "case_id": "1245-2024",
  "template_name": "Klagebegründung_Vorlage.odt",
  "file_name": "Klagebegründung_2024-05-05.odt",
  "generated_text": "[Text aus /generate]"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Vorlage erfolgreich an j-lawyer gesendet"
}
```

**Request (`GET /jlawyer/templates`):**
```
GET /jlawyer/templates
```

**Response:**
```json
{
  "templates": ["Klagebegründung_Vorlage.odt", "Schriftsatz_Standard.odt"],
  "folder": "Klagebegründungen"
}
```

### Processing Flow

1. **Input Validation**
   - Check rate limits
   - Validate document selections
   - Ensure at least one Bescheid is marked as primary

2. **Document Collection**
   - Query database for selected documents
   - Load PDFs from filesystem:
     - `/app/uploads/` for Anhörung, Bescheid, Rechtsprechung
     - `/app/downloaded_sources/` for saved sources
   - Extract metadata (dates, Az., etc.) from database records
   - Upload each PDF with the Claude Files API and capture resulting `file_id`s

3. **Prompt Construction**
   - Build comprehensive prompt with:
     - Task definition (Klagebegründung vs Schriftsatz)
     - User instructions
     - Document context (labeled by category)
     - Citation requirements
     - Structure guidance

4. **Claude API Call**
   - Model: `claude-sonnet-4-5`
   - Call `client.beta.messages.create` with `betas=["files-api-2025-04-14"]`
   - Reference each uploaded PDF via `{"type": "document", "source": {"type": "file", "file_id": "..."}}`
   - Request structured output with proper citations

5. **Output Validation**
   - **Citation Verification**: Check that provided sources are referenced in output
   - Log warnings if sources are unused

6. **Return Result**
   - Return structured response with `generated_text`, `used_documents`, and `metadata`
   - Surface citation warnings in logs/UI (non-blocking)

## Citation Style Requirements

### Numbered Anlagen

- **Vollmacht**: Always referenced as "Anlage K1" (document not uploaded via this system)
- **Primary Bescheid**: "Anlage K2"
  - Full reference: "Bescheid vom [Datum], Az. [Aktenzeichen], Anlage K2"

### Non-numbered Citations

- **Anhörung**: "Anhörung vom [Datum]"
  - No Anlage number
  - No Az. (typically not present in Anhörung documents)

- **Additional Bescheide** (not primary): "Bescheid vom [Datum], Az. [Aktenzeichen]"
  - No Anlage number for secondary Bescheide

- **Rechtsprechung**: "[Gericht], Az. [Aktenzeichen], Urteil vom [Datum], [Link]"
  - Example: "VG München, Az. M 1 K 18.12345, Urteil vom 15.03.2023, https://..."

- **asyl.net Sources**: "[Titel/Gericht], Az. [falls vorhanden], [Datum], [Link]"
  - Example: "BVerwG zu Abschiebungsverbot Syrien, Az. 1 C 16.22, 12.01.2023, https://asyl.net/..."

- **Web Research Sources**: "[Titel], [Datum], [Link]"
  - Example: "BAMF Herkunftsländerinformation Syrien, 20.02.2023, https://..."

## Document Structure & Style

### Structure

Generated documents should follow this structure:

```
I. Sachverhalt

[Fließtext describing the case background, referencing Anhörung and Bescheid]

II. Rechtliche Würdigung

[Fließtext with legal analysis, referencing case law and legal sources]
```

**Note:** No "III. Antrag" section - this will be provided by j-lawyer template.

### Style Requirements

- **Tone**: Formal legal German (Fachanwalt für Ausländerrecht)
- **Format**: Fließtext (continuous text, not bullet points)
- **Length**: Detailed but concise (avoid verbosity)
- **Language**: Professional legal terminology appropriate for court submissions

### Claude Prompt Guidelines

The prompt to Claude should specify:

1. Role: "Sie sind ein Fachanwalt für Ausländerrecht in Deutschland"
2. Task: Generate Klagebegründung/Schriftsatz
3. Structure: I. Sachverhalt, II. Rechtliche Würdigung
4. Citation requirements (as documented above)
5. User-specific instructions from prompt
6. All attached documents with clear labels

## Frontend UI

### Document Selection Interface

Create a section in the UI with:

**Anhörung:**
- [ ] anhoerung1.pdf (12.03.2023)
- [ ] anhoerung2.pdf (15.04.2023)

**Bescheid:**
- [ ] bescheid1.pdf (20.05.2023) [⭐ Als Hauptbescheid markieren]
- [ ] bescheid2.pdf (10.06.2023) [⭐ Als Hauptbescheid markieren]

**Rechtsprechung:**
- [ ] urteil1.pdf (VG München, 2023)
- [ ] urteil2.pdf (BVerwG, 2022)

**Gespeicherte Quellen:**
- [ ] asyl.net: BVerwG zu Abschiebungsverbot
- [ ] BAMF Herkunftsländerinformation

**Prompt:**
```
[Textarea für Benutzeranweisungen]
Beispiel: "Beantrage Feststellung eines Abschiebungsverbots nach § 60 Abs. 5 AufenthG wegen psychischer Erkrankung. Schwerpunkt auf individuelle Gefährdung legen."
```

**Button:** "Klagebegründung generieren"

### Preview Modal

After generation, show modal with:

- **Editable textarea** with generated text
- **Metadata display**: Documents used, citations found
- **Action buttons**:
  - "Bearbeiten" (already editable)
  - "An j-lawyer senden"
  - "Als Text speichern" (download .txt)
  - "Schließen"

## j-lawyer Integration

### API Endpoint

**Endpoint:** `PUT /v6/templates/documents/{folder}/{template}/{caseId}/{fileName}`

**Authentication:** HTTP Basic Auth with j-lawyer credentials

**Request Body:**
```json
[
  {
    "placeHolderKey": "HAUPTTEXT",
    "placeHolderValue": "[Generated text from Claude]"
  }
]
```

### Configuration

The system needs to store j-lawyer configuration:

**Environment Variables:**
```bash
JLAWYER_BASE_URL=http://localhost:8080/j-lawyer-io
JLAWYER_USERNAME=admin
JLAWYER_PASSWORD=password
JLAWYER_TEMPLATE_FOLDER=Klagebegründungen
JLAWYER_PLACEHOLDER_KEY=HAUPTTEXT
```

### j-lawyer API Call Flow

1. User clicks "An j-lawyer senden"
2. Frontend sends: `POST /send-to-jlawyer`
   ```json
   {
     "case_id": "12345",
     "template_name": "Klagebegründung_Vorlage.odt",
     "file_name": "Klagebegründung_[Client]_[Date].odt",
     "generated_text": "[text from modal]"
   }
   ```
3. Backend calls j-lawyer API
4. Return success/error to frontend

The frontend fetches available template names via `GET /jlawyer/templates` once per session and pre-populates a dropdown in the modal. Users can either pick one of the discovered templates or provide an alternative name manually; the backend will use whichever value is supplied in `template_name`.

### Template Preparation

User must create ODT template in j-lawyer with placeholder:

```
{{HAUPTTEXT}}
```

This placeholder will be replaced with the generated text.

## Validation Requirements

### Citation Verification

After generation, verify that:

1. **All provided sources are cited**:
   - Check if each selected document appears in generated text
   - Log warnings for unused sources (but don't block generation)

2. **Citation format is correct**:
   - Anlage K1, K2 used appropriately
   - Dates and Az. present where expected
   - Links included for Rechtsprechung and web sources

### Implementation

- Normalise the generated text and compare it against candidate strings derived from filenames, titles, URLs, and explanations of each selected document.
- Treat missing matches as non-blocking warnings; log and return them in `metadata.missing_citations` so the UI can surface follow-up work.
- For the primary Bescheid enforce an `Anlage K2` check, adding a warning if the phrase is absent even when the document is mentioned.

## Database Schema Extensions

No new tables required. Use existing:

- `documents` table: For Anhörung, Bescheid, Rechtsprechung
- `research_sources` table: For gespeicherte Quellen

## Error Handling

### Common Error Scenarios

1. **No Bescheid marked as primary**:
   - Return 400: "Bitte markieren Sie einen Bescheid als Hauptbescheid (Anlage K2)"

2. **No documents selected**:
   - Return 400: "Bitte wählen Sie mindestens ein Dokument aus"

3. **Claude API error**:
   - Return 500: "Fehler bei der Textgenerierung. Bitte versuchen Sie es erneut."
   - Log full error for debugging

4. **j-lawyer API error**:
   - Return 502: "Fehler beim Senden an j-lawyer: [error message]"
   - Allow user to download text manually

5. **Rate limit exceeded**:
   - Return 429: "Rate limit erreicht. Bitte warten Sie [X] Minuten."

## Future Enhancements

- **Multiple document types**: Support for Widerspruch, Eilantrag, etc.
- **Batch generation**: Generate multiple documents at once
- **Version history**: Track iterations of generated documents
- **Quality scoring**: Rate output quality and provide feedback to Claude
- **Export formats**: Direct ODT/PDF export without j-lawyer

## Implementation Checklist

- [x] Backend: Extend `/generate` endpoint for Klagebegründung flow
- [x] Backend: Implement document collection logic
- [x] Backend: Design Claude prompt with citation requirements
- [x] Backend: Implement Claude API integration
- [x] Backend: Implement citation verification
- [x] Backend: Create j-lawyer API integration
- [x] Frontend: Build document selection UI
- [x] Frontend: Add Bescheid primary marking
- [x] Frontend: Create preview modal
- [x] Frontend: Implement "Send to j-lawyer" flow
- [ ] Testing: Test with real documents
- [x] Testing: Validate citation quality
- [x] Documentation: Update CLAUDE.md with new feature
- [x] Configuration: Add j-lawyer environment variables

## Testing Strategy

### Unit Tests

- Document collection logic
- Citation verification algorithm
- Metadata extraction

### Integration Tests

- End-to-end generation flow
- j-lawyer API interaction (with mock server)

### Manual Testing Scenarios

1. **Basic generation**: 1 Anhörung, 1 Bescheid, 1 Rechtsprechung
2. **Multiple documents**: 3+ documents per category
3. **Citation verification**: Ensure all sources are referenced
4. **Edit and resend**: Modify text and send to j-lawyer
5. **Error scenarios**: Test all error cases listed above

## Performance Considerations

- **Document limit**: No hard limit, but warn if >20 documents total
- **PDF size**: Claude has token limits; log warning if documents are very large
- **Generation time**: Expected 30-60 seconds for typical case
- **Caching**: Consider caching PDF encodings for frequently used documents

## Security Considerations

- **Rate limiting**: Strict limits to prevent abuse (10/hour)
- **Input validation**: Sanitize all user inputs
- **j-lawyer credentials**: Store securely in environment variables
- **Access control**: Future feature - ensure users can only access their own cases
