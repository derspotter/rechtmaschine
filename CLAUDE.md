# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rechtmaschine is an AI-powered legal document classification and generation tool for German asylum lawyers. The project is deployed on a self-hosted server behind a Caddy reverse proxy at `rechtmaschine.de`.

**Current Status:** MVP phase with working document classification. Initial document generation via Claude is implemented.

## Architecture

### Technology Stack
- **Backend:** FastAPI (Python 3.11) running in Docker
- **AI:** OpenAI GPT-5-mini via Responses API with structured output
- **Frontend:** Embedded HTML/CSS/JS (no separate framework yet)
- **Reverse Proxy:** Caddy (HTTPS with automatic certificates)
- **Storage:** File-based JSON at `/app/classifications.json` (MVP approach)
- **Uploads:** Persisted under `/app/uploads` and linked from classification entries (`file_path`)
- **Deployment:** Docker Compose on self-hosted server

### Document Classification Flow
1. User uploads PDF via web interface
2. PDF sent to `/classify` endpoint
3. File uploaded to OpenAI using `client.files.create()`
4. Classification performed via OpenAI Responses API (`client.responses.parse()`)
   - Model: `gpt-5-mini`
   - Uses structured output with Pydantic models
   - Service tier: `flex` for cost optimization
5. Result saved to JSON file
6. Frontend displays document in appropriate category box

### Document Generation Flow (New)
1. User enters a description and clicks “Entwurf generieren”
2. Backend `/generate` collects up to 4 recent uploaded PDFs as context
3. PDFs are attached to Claude (`claude-3-5-sonnet-20241022`) via message attachments
4. Draft text is returned and shown in a modal; drafts are not persisted yet

### Key Implementation Details

**OpenAI Responses API Pattern:**
The code uses OpenAI's Responses API (not Chat Completions) with the following structure:
```python
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
response = client.with_options(timeout=900.0).responses.parse(**request_params)
parsed_result = response.output_parsed
```

**Document Categories:**
- Anhörung (hearing protocols)
- Bescheid (administrative decisions)
- Rechtsprechung (case law)
- Sonstiges (other)

**Environment Variables:**
- `OPENAI_API_KEY` - Required for classification
- `ANTHROPIC_API_KEY` - Required for document generation (Claude)
- `GOOGLE_API_KEY` - For future web search (Gemini)

## Development Commands

### Running the Application

**Start the app:**
```bash
cd /var/opt/docker/rechtmaschine/app
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f app
```

**Restart after code changes:**
```bash
docker-compose restart app
```

**Note:** Hot reload is configured but doesn't work reliably with Docker volume mounts. Manual restart is required after changes to `app.py`.

### Caddy Reverse Proxy

**Update Caddy configuration:**
After editing `/var/opt/docker/caddy/Caddyfile`, restart Caddy:
```bash
docker restart caddy
```

**Note:** `caddy reload` doesn't work reliably for bind-mounted config files. Always use `docker restart caddy`.

### Accessing the Application
- Main app: https://rechtmaschine.de
- n8n instance: https://n8n.rechtmaschine.de (configured but not actively used)

## File Structure

```
/var/opt/docker/rechtmaschine/app/
├── app.py              # Main FastAPI application (all backend + frontend HTML)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose configuration
├── .env               # API keys (not in git)
└── .env.example       # Example environment variables
```

Uploads are stored inside the container at: `/app/uploads/`.

## Important Constraints

### Docker Network
The app connects to the external `caddy` network (not `caddy_default`). The Caddy reverse proxy is defined elsewhere on the server.

### Dependencies
- `openai>=1.56.0` - Required for Responses API support
- `httpx` - HTTP client (see version in requirements.txt)
- `pikepdf==9.4.2` - For PDF text extraction (currently not used in classification)
- `google-genai==1.41.0` - Gemini + Google Search grounding
- `anthropic` - Claude client used by `/generate`

### German Language Support
All UI text and prompts are in German. Category names use German umlauts which must be mapped correctly in JavaScript:
```javascript
const categoryMap = {
    'Anhörung': 'anhoerung-docs',
    'Bescheid': 'bescheid-docs',
    'Rechtsprechung': 'rechtsprechung-docs',
    'Sonstiges': 'sonstiges-docs'
};
```

## Future Development

See `plan.md` for the full roadmap. Key planned features:
- Document generation using Claude (initial version done; add persistence + ODT/PDF export)
- Web search integration with Gemini
- Multi-user authentication
- Migration from file-based to database storage
- Svelte frontend (currently embedded HTML)
- PDF/ODT output with LibreOffice templates

## Reference Implementation

The classification approach is based on `/home/jay/GHPL/meta_ghpl_gpt5.py` (outside this repo). When modifying classification logic, maintain alignment with that reference pattern.
