# TODO: Rechtsprechung Playbook for Generate Endpoint

Goal
- Build a persistent "Aktuelle Rechtsprechung" knowledge base stored in the DB.
- Entries are created on-demand (not for every upload) and enriched via LLM extraction.
- All users can add entries and edit tags; show only countries that have entries.
- Generation integration is OUT OF SCOPE for now.

Plan
1) Define DB schema for "Aktuelle Rechtsprechung"
- New table (e.g., `rechtsprechung_entries`) with: country, tags (array), court, date, aktenzeichen, outcome, key holdings, key facts, citations, source_document_id, created_at/updated_at, is_active.
- Tags are editable by users (LLM-generated initial tags + manual edits).
- Store a short summary for UI display.

2) Add on-demand extraction flow
- Add UI action: "➕ In Aktuelle Rechtsprechung übernehmen".
- Only then call Gemini 3 Flash (structured output) to extract metadata + tags.
- Deduplicate by court+date+AZ (update existing entry if duplicate).

3) Backend endpoints
- CRUD endpoints for entries and tags (list/filter/update/delete).
- Allow any authenticated user to edit tags.
- Return only countries with existing entries for filter UI.

4) UI repurpose (Rechtsprechung box)
- Split view: "Dokumente (Rechtsprechung)" + "Aktuelle Rechtsprechung".
- Show richer metadata (country, date, court, AZ, tags, short holdings).
- Filters by country (only countries with entries) + tag chips.

5) Provenance & freshness (optional)
- Show last updated per entry.
- Optional warning if entry is older than N days.

6) Permissions
- All users can add entries and edit tags.
- No admin-only restriction needed for now.

7) Safety and accuracy
- Require citations in the playbook and include them in the generated output.
- Add a disclaimer in the prompt to avoid inventing jurisprudence.

8) Manual validation
- Test a few common case types to verify the correct argumentation patterns appear.
- Verify that the endpoint still works with empty or missing playbook.
