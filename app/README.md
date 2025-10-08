# Rechtmaschine Document Classifier

A simple document classification system for German asylum law documents.

## Categories

The classifier can categorize documents into 4 types:

1. **Anhörung** - Hearing protocols from BAMF asylum hearings
2. **Bescheid** - Administrative decisions/rulings from BAMF
3. **Rechtsprechung** - Court decisions and case law
4. **Sonstiges** - Other documents

## Setup

1. Copy `.env.example` to `.env` and add your OpenAI API key:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

2. Build and start the application:
   ```bash
   docker-compose up -d --build
   ```

3. Access the web interface at `http://localhost:8000`

## Usage

1. Open the web interface in your browser
2. Select a PDF file to upload
3. Click "Dokument klassifizieren"
4. View the classification result with confidence score and explanation

## API Endpoints

- `GET /` - Web interface
- `POST /classify` - Classify a document (accepts PDF file upload)
- `GET /health` - Health check

## API Example

```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "category": "Bescheid",
  "confidence": 0.95,
  "explanation": "Dies ist ein BAMF-Bescheid, erkennbar an der offiziellen Struktur und den Verfügungssätzen.",
  "filename": "document.pdf"
}
```
