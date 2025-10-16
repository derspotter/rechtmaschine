# Rechtmaschine

AI-powered legal document classification, research, and generation tool for German asylum law.

## Overview

Rechtmaschine assists German asylum lawyers by automatically classifying legal documents, conducting intelligent web research, and generating legal drafts. The system leverages multiple AI models and features automatic document segmentation for complex case files.

## Tech Stack

- **Frontend:** Embedded HTML/CSS/JS (Svelte migration planned)
- **Backend:** FastAPI (Python 3.11) + PostgreSQL
- **AI Models:**
  - Gemini 2.5 Flash (classification & segmentation)
  - Gemini 2.5 Flash (web research with Google Search grounding)
  - Claude 3.5 Sonnet (document generation)
- **Web Scraping:** Playwright (asyl.net integration & PDF detection)
- **Deployment:** Docker Compose with Caddy reverse proxy

## Features

- **Intelligent Document Classification**: Automatically categorizes uploaded PDFs into Anhörung, Bescheid, Akte, Rechtsprechung, or Sonstiges
- **Automatic PDF Segmentation**: When complete case files (Akte) are uploaded, automatically extracts individual documents
- **Web Research**: Dual search combining Gemini with Google Search grounding and asyl.net legal database
- **Saved Sources Management**: Download and organize legal research sources with real-time status updates
- **Draft Generation**: Generate legal documents using Claude with context from uploaded PDFs
- **Multi-user Support**: PostgreSQL-backed persistent storage for documents and sources

## Getting Started

See [plan.md](plan.md) for detailed project planning and architecture.

## Development

This project is developed directly on a self-hosted server for security and deployment efficiency.

---

🤖 Generated with Claude Code
