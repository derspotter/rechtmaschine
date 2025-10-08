"""
Rechtmaschine - Document Classifier
Simplified document classification system for German asylum law documents
"""

import os
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
import tempfile
import pikepdf
from openai import OpenAI
from enum import Enum

app = FastAPI(title="Rechtmaschine Document Classifier")

# Storage file path
CLASSIFICATIONS_FILE = Path("/app/classifications.json")

# Document categories
class DocumentCategory(str, Enum):
    ANHOERUNG = "Anhörung"  # Hearing protocols
    BESCHEID = "Bescheid"  # Administrative decisions/rulings
    RECHTSPRECHUNG = "Rechtsprechung"  # Case law
    SONSTIGES = "Sonstiges"  # Other

class ClassificationResult(BaseModel):
    category: DocumentCategory
    confidence: float
    explanation: str
    filename: str

# Storage functions
def load_classifications() -> List[Dict]:
    """Load classifications from JSON file"""
    if not CLASSIFICATIONS_FILE.exists():
        return []
    try:
        with open(CLASSIFICATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading classifications: {e}")
        return []

def save_classification(result: ClassificationResult) -> None:
    """Save a classification result to JSON file"""
    classifications = load_classifications()

    # Remove existing entry for same filename if exists
    classifications = [c for c in classifications if c['filename'] != result.filename]

    # Add new classification
    classifications.append({
        'filename': result.filename,
        'category': result.category.value,
        'confidence': result.confidence,
        'explanation': result.explanation,
        'timestamp': datetime.now().isoformat()
    })

    try:
        with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving classification: {e}")

def delete_classification(filename: str) -> bool:
    """Delete a classification by filename"""
    classifications = load_classifications()
    original_length = len(classifications)
    classifications = [c for c in classifications if c['filename'] != filename]

    if len(classifications) == original_length:
        return False  # Nothing was deleted

    try:
        with open(CLASSIFICATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(classifications, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error deleting classification: {e}")
        return False

# Initialize OpenAI client
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

def extract_pdf_text(pdf_path: str, max_pages: int = 5) -> str:
    """Extract text from first few pages of PDF"""
    try:
        with pikepdf.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = min(total_pages, max_pages)

            text_parts = []
            for i in range(pages_to_read):
                # Extract text from page
                page = pdf.pages[i]
                if hasattr(page, 'extract_text'):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- Page {i+1} ---\n{text}")

            return "\n\n".join(text_parts) if text_parts else ""
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")

async def classify_document(file_content: bytes, filename: str) -> ClassificationResult:
    """Classify document using OpenAI Responses API with GPT-5-mini"""

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    try:
        client = get_openai_client()

        # Upload PDF to OpenAI
        with open(tmp_path, "rb") as f:
            uploaded_file = client.files.create(file=f, purpose="user_data")

        try:
            # Build prompt for classification
            prompt = """Klassifiziere dieses deutsche Rechtsdokument in eine der folgenden Kategorien:

1. **Anhörung** - Anhörungsprotokolle vom BAMF
   - Merkmale: Frage-Antwort-Format, Dolmetscher, persönliche Geschichte des Antragstellers

2. **Bescheid** - BAMF-Bescheide über Asylanträge
   - Merkmale: Offizieller BAMF-Briefkopf, Verfügungssätze, Rechtsbehelfsbelehrung

3. **Rechtsprechung** - Gerichtsentscheidungen, Urteile
   - Merkmale: Gericht als Absender, Aktenzeichen, Tenor, Tatbestand, Entscheidungsgründe

4. **Sonstiges** - Andere Dokumente

Gib deine Antwort mit category (eine der vier Kategorien), confidence (0.0-1.0) und explanation (kurze Begründung auf Deutsch) zurück."""

            # Build request parameters for Responses API (matching meta_ghpl_gpt5.py)
            request_params = {
                "model": "gpt-5-mini",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_id": uploaded_file.id},
                            {"type": "input_text", "text": prompt}
                        ]
                    }
                ],
                "text_format": ClassificationResult,
                "service_tier": "flex"
            }

            # Use Responses API with structured output (matching meta_ghpl_gpt5.py pattern)
            response = client.with_options(timeout=900.0).responses.parse(**request_params)

            # Clean up uploaded file
            client.files.delete(uploaded_file.id)

            # Extract the parsed result from ParsedResponse
            parsed_result = response.output_parsed

            # Create ClassificationResult with filename
            return ClassificationResult(
                category=parsed_result.category,
                confidence=parsed_result.confidence,
                explanation=parsed_result.explanation,
                filename=filename
            )

        except Exception as e:
            # Clean up uploaded file in case of error
            try:
                client.files.delete(uploaded_file.id)
            except:
                pass
            raise e

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve document classification interface with category boxes"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rechtmaschine - Document Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f5f5;
                padding: 20px;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .upload-section {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .btn {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            .btn:hover {
                background-color: #2980b9;
            }
            .loading {
                display: none;
                color: #7f8c8d;
                font-style: italic;
                text-align: center;
                margin: 10px 0;
            }
            .categories-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .category-box {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                min-height: 300px;
            }
            .category-box h3 {
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 3px solid;
            }
            .category-box.anhoerung {
                border-top: 4px solid #3498db;
            }
            .category-box.anhoerung h3 {
                color: #3498db;
                border-color: #3498db;
            }
            .category-box.bescheid {
                border-top: 4px solid #27ae60;
            }
            .category-box.bescheid h3 {
                color: #27ae60;
                border-color: #27ae60;
            }
            .category-box.rechtsprechung {
                border-top: 4px solid #e67e22;
            }
            .category-box.rechtsprechung h3 {
                color: #e67e22;
                border-color: #e67e22;
            }
            .category-box.sonstiges {
                border-top: 4px solid #95a5a6;
            }
            .category-box.sonstiges h3 {
                color: #95a5a6;
                border-color: #95a5a6;
            }
            .document-card {
                background: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 3px solid #3498db;
                position: relative;
            }
            .document-card .filename {
                font-weight: bold;
                color: #2c3e50;
                word-break: break-word;
                margin-bottom: 5px;
            }
            .document-card .confidence {
                font-size: 12px;
                color: #7f8c8d;
            }
            .document-card .delete-btn {
                position: absolute;
                top: 5px;
                right: 5px;
                background: #e74c3c;
                color: white;
                border: none;
                border-radius: 3px;
                width: 20px;
                height: 20px;
                cursor: pointer;
                font-size: 12px;
                line-height: 1;
            }
            .document-card .delete-btn:hover {
                background: #c0392b;
            }
            .empty-message {
                color: #95a5a6;
                font-style: italic;
                text-align: center;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏛️ Rechtmaschine</h1>
            <p>Dokumenten-Klassifikation für Asylrecht</p>
        </div>

        <div class="upload-section">
            <h3>Dokument hochladen und klassifizieren</h3>
            <input type="file" id="fileInput" accept=".pdf" />
            <br>
            <button class="btn" onclick="uploadFile()">Dokument klassifizieren</button>
            <div class="loading" id="loading">⏳ Dokument wird analysiert...</div>
        </div>

        <div class="upload-section">
            <h3>Rechtsdokument generieren</h3>
            <p style="color: #7f8c8d; font-size: 14px; margin-bottom: 10px;">
                Beschreiben Sie, welches Dokument Sie generieren möchten (z.B. Klagebegründung, Antrag, etc.)
            </p>
            <textarea id="outputDescription"
                      placeholder="Beispiel: Ich benötige eine Klagebegründung für einen Asylantrag aus Syrien..."
                      style="width: 100%; min-height: 100px; padding: 10px; border: 1px solid #bdc3c7; border-radius: 5px; font-family: Arial; margin-bottom: 10px;">
            </textarea>
            <button class="btn" onclick="generateDocument()" style="background-color: #27ae60;">
                Dokument generieren (kommt bald)
            </button>
        </div>

        <div class="categories-grid">
            <div class="category-box anhoerung">
                <h3>📋 Anhörung</h3>
                <div id="anhoerung-docs"></div>
            </div>
            <div class="category-box bescheid">
                <h3>📄 Bescheid</h3>
                <div id="bescheid-docs"></div>
            </div>
            <div class="category-box rechtsprechung">
                <h3>⚖️ Rechtsprechung</h3>
                <div id="rechtsprechung-docs"></div>
            </div>
            <div class="category-box sonstiges">
                <h3>📁 Sonstiges</h3>
                <div id="sonstiges-docs"></div>
            </div>
        </div>

        <script>
            // Load documents on page load
            window.addEventListener('DOMContentLoaded', loadDocuments);

            async function loadDocuments() {
                try {
                    const response = await fetch('/documents');
                    const data = await response.json();

                    // Clear all boxes
                    document.getElementById('anhoerung-docs').innerHTML = '';
                    document.getElementById('bescheid-docs').innerHTML = '';
                    document.getElementById('rechtsprechung-docs').innerHTML = '';
                    document.getElementById('sonstiges-docs').innerHTML = '';

                    // Populate each category
                    const categoryMap = {
                        'Anhörung': 'anhoerung-docs',
                        'Bescheid': 'bescheid-docs',
                        'Rechtsprechung': 'rechtsprechung-docs',
                        'Sonstiges': 'sonstiges-docs'
                    };

                    for (const [category, documents] of Object.entries(data)) {
                        const boxId = categoryMap[category];
                        const box = document.getElementById(boxId);

                        if (box) {
                            if (documents.length === 0) {
                                box.innerHTML = '<div class="empty-message">Keine Dokumente</div>';
                            } else {
                                box.innerHTML = documents.map(doc => createDocumentCard(doc)).join('');
                            }
                        }
                    }
                } catch (error) {
                    console.error('Error loading documents:', error);
                }
            }

            function createDocumentCard(doc) {
                const confidence = (doc.confidence * 100).toFixed(0);
                const escapedFilename = doc.filename.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                return `
                    <div class="document-card">
                        <button class="delete-btn" onclick="deleteDocument('${escapedFilename}')" title="Löschen">×</button>
                        <div class="filename">${doc.filename}</div>
                        <div class="confidence">${confidence}% Konfidenz</div>
                    </div>
                `;
            }

            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];

                if (!file) {
                    alert('Bitte wählen Sie eine PDF-Datei aus');
                    return;
                }

                const loading = document.getElementById('loading');
                loading.style.display = 'block';

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/classify', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();

                    if (response.ok) {
                        // Reload documents to show the new one
                        await loadDocuments();
                        // Clear file input
                        fileInput.value = '';
                        // Show success message
                        alert(`✅ Dokument klassifiziert als: ${data.category} (${(data.confidence * 100).toFixed(0)}%)`);
                    } else {
                        alert(`❌ Fehler: ${data.detail || 'Unbekannter Fehler'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function deleteDocument(filename) {
                if (!confirm(`Möchten Sie "${filename}" wirklich löschen?`)) {
                    return;
                }

                try {
                    const response = await fetch(`/documents/${encodeURIComponent(filename)}`, {
                        method: 'DELETE'
                    });

                    if (response.ok) {
                        // Reload documents
                        await loadDocuments();
                    } else {
                        const data = await response.json();
                        alert(`❌ Fehler: ${data.detail || 'Löschen fehlgeschlagen'}`);
                    }
                } catch (error) {
                    alert(`❌ Fehler: ${error.message}`);
                }
            }

            function generateDocument() {
                const description = document.getElementById('outputDescription').value.trim();

                if (!description) {
                    alert('Bitte beschreiben Sie das gewünschte Dokument');
                    return;
                }

                // Placeholder for future implementation
                alert('📝 Dokumentgenerierung kommt in der nächsten Version!\\n\\n' +
                      'Diese Funktion wird die klassifizierten Dokumente als Kontext verwenden, um rechtliche Texte zu generieren.');
            }
        </script>
    </body>
    </html>
    """

@app.post("/classify", response_model=ClassificationResult)
async def classify(file: UploadFile = File(...)):
    """Classify uploaded PDF document"""

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    # Classify document
    try:
        result = await classify_document(content, file.filename)
        # Save classification to storage
        save_classification(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.get("/documents")
async def get_documents():
    """Get all classified documents grouped by category"""
    classifications = load_classifications()

    # Group by category
    grouped = {
        "Anhörung": [],
        "Bescheid": [],
        "Rechtsprechung": [],
        "Sonstiges": []
    }

    for classification in classifications:
        category = classification.get('category', 'Sonstiges')
        if category in grouped:
            grouped[category].append(classification)

    return grouped

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a classified document"""
    success = delete_classification(filename)
    if success:
        return {"message": f"Document {filename} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Document {filename} not found")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
