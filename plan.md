# Rechtmaschine - AI Agent for German Asylum Law

## Project Overview
An AI-powered tool to assist German asylum lawyers in generating legal documents and texts.

## Core Functionality
- Generate legal texts including:
  - Klagebegründungen (lawsuit justifications)
  - Anträge (applications)
  - Other legal documents
- Accept multiple input document types as context
- Perform intelligent web searches for additional context
- Format output according to pre-existing stylesheets

## User Interface
- Simple GUI for document input
- Supported input document types:
  - Anhörung (hearing protocols)
  - Bescheid (administrative decisions/rulings)
  - Rechtsprechung (case law)
  - Ähnliche Fälle (similar cases)
  - [More types to be defined]
- Text field for user to describe desired output
- [UI mockup/design TBD]

## Technical Architecture

### Technology Stack
- **Frontend:** Svelte ✓
- **Backend/Orchestration:** n8n (self-hosted) + Python for document processing ✓
- **AI/LLM:** Multi-model approach ✓
  - Gemini - Web search and information retrieval
  - Claude - Primary text generation
  - ChatGPT - Error checking (logical inconsistencies, legal accuracy)
- **Document Processing:** Python scripts (called from n8n) ✓
  - Input formats: PDF (primary), DOCX, scanned images, URLs/links
  - Document sizes: Up to 50MB, typically <10MB
  - Processing approach: Full-text context (no structured extraction)
  - Scanned documents: Direct upload to multimodal AI (Claude/Gemini/GPT-4 Vision) ✓
- **Web Search:** Gemini API with Google Search Grounding ✓
  - Method: Gemini's built-in `google_search` tool
  - Features: Automatic query generation, inline source citations, real-time web content
  - Targets: General web (via Google Search) + specific German legal databases
  - Cost: $35 per 1,000 grounded queries (requires paid tier)
  - Additional searches: May need separate API calls for specific legal databases
- **Output Formats:** PDF and ODT ✓
- **Stylesheet:** LibreOffice template (.odt or .ott file) ✓
- **Citations:** Footnotes format ✓
- **User Management:** Multi-user support ✓
- **Deployment:** Self-hosted (for client confidentiality) ✓

### Document Processing Pipeline
1. User uploads documents (Anhörung, Bescheid, Rechtsprechung, etc.)
2. System parses and extracts relevant information
3. User provides description of desired output
4. AI agent performs web search for additional context
5. AI generates text using:
   - Uploaded documents
   - Web search results
   - User description
   - Pre-existing style guidelines
6. Output is formatted according to stylesheet
7. User receives formatted document

## Key Questions to Resolve

### Technical Decisions
- [x] Frontend technology choice - Svelte
- [x] Backend framework - n8n (self-hosted) + Python for document processing
- [x] LLM provider and model - Multi-model (Gemini, Claude, ChatGPT)
- [x] Workflow orchestration - n8n with visual workflows
- [x] Deployment environment - Self-hosted
- [x] Document input formats - PDF (primary), DOCX, scanned images, URLs
- [x] Document size handling - Up to 50MB, typically <10MB
- [x] Output formats - PDF and ODT
- [x] Stylesheet format - LibreOffice template (.odt or .ott)
- [x] Web search targets - General web + German legal databases
- [x] OCR approach - Multimodal AI (no separate OCR needed)
- [x] Gemini search method - Google Search Grounding ($35/1k queries)

### Functional Requirements
- [x] Document parsing approach - Full-text context (no structured extraction)
- [x] Web search target sources - Both general web and German legal databases
- [x] Citation and source attribution handling - Footnotes format
- [x] User authentication requirements - Multi-user support needed
- [x] Specific error types for ChatGPT to check - Logical inconsistencies, legal accuracy
- [ ] LLM pipeline flow (sequential vs parallel)
- [ ] Authentication method (OAuth, email/password, LDAP?)

### Privacy & Security
- [ ] Data storage location and security
- [ ] Client confidentiality measures
- [ ] GDPR compliance considerations
- [ ] Access control requirements

## Development Strategy: Server vs Local

### Direct Server Development (Recommended ✓)

**Pros:**
- n8n already installed on server (or install once, use always)
- Faster deployment - no separate deployment step
- Test in production-like environment from day one
- Client confidentiality from the start
- No localhost → server migration issues
- Easier to test multi-user features
- LibreOffice conversion works in real environment

**Cons:**
- Need SSH/remote development setup
- Slightly slower feedback loop during development
- Need good internet connection

**Recommended Approach:**
1. Develop on server with VS Code Remote SSH (or similar)
2. Use git for version control
3. Set up development and production n8n instances on same server
4. Svelte with hot-reload still works over SSH

### Alternative: Local Development

**Pros:**
- Faster iteration during UI development
- No internet needed for development

**Cons:**
- Need to install/configure n8n locally
- Migration to server later
- Different environments = potential bugs

**Decision:** Develop directly on server ✓

## Next Steps
1. Set up development environment on server
2. Install n8n on server
3. Create detailed technical architecture
4. Define data models and document schemas
5. Design UI mockups
6. Plan implementation phases

## n8n Integration Discussion

### Potential Use Cases for n8n:
n8n could be excellent for orchestrating the multi-LLM workflow:

**What n8n could handle:**
- Document processing pipeline orchestration
- Sequential API calls to Gemini → Claude → ChatGPT
- Conditional logic (e.g., if Gemini finds X, then do Y)
- Web search integration
- Error handling and retries for API calls
- Webhooks to trigger workflows from your Svelte app

**Architecture Options:**

**Option A: n8n as Backend Orchestration Layer**
- Svelte frontend → n8n webhooks → LLM pipeline → response
- Pros: Visual workflow design, easy to modify pipeline, built-in nodes for LLMs
- Cons: Less code control, learning curve, potential vendor lock-in

**Option B: Python Backend with n8n for Complex Workflows**
- Python FastAPI handles core logic, n8n for specific multi-step workflows
- Hybrid approach for best of both worlds

**Option C: Pure Python Backend**
- Full control, easier version control, more flexible
- Write orchestration logic in Python using libraries like LangChain or custom code

### Questions to Decide:
- Do you want visual workflow design vs. code-first approach?
- Is the pipeline relatively fixed or will it change frequently?
- Do you need version control for the workflow logic?
- Self-hosted n8n or n8n cloud?

## Notes
- Project name: "rechtmaschine" (law machine)
- Target users: German asylum lawyers
- Primary goal: Increase efficiency in legal document preparation
