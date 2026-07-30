# RAG-Frontend „Wissensbasis" — Design

**Datum:** 2026-07-30 · **Status:** entworfen, von Jay mündlich freigegeben (Belegpflicht + Chat-Persistenz nachgezogen)

## Ziel

Die RAG-Infrastruktur (debian-Store, `/v1/rag/*`-Proxy, `rag_context`) ist im Frontend
unsichtbar. Neuer Haupt-Tab „📚 Wissensbasis" macht sie direkt nutzbar: Treffersuche
über den Bestand **und** ein Chat, der Fragen mit LLM-Hilfe beantwortet — immer
belegt und zitiert, mit persistenten Chats.

## Entscheidungen (Jay, 30.07.2026)

- Suche **und** Chat (ein Tab, ein Eingabefeld, zwei Buttons „Suchen"/„Fragen").
- Collections: alle drei wählbar — `kanzlei`, `jurisprudence`, `doktrin` — Default „Alle".
- Chat-LLM: Modell-Dropdown wie im Befragen-Feld (Gemini 3.6 Flash default, Gemini 3.1 Pro,
  GPT-5.6 Terra, Claude Sonnet 5, Grok 4.5). Chunks sind anonymisiert/öffentlich → Cloud ok.
- **Belegpflicht:** jede Aussage der Chat-Antwort mit Quellen-Marker, fail-closed.
- **Chat-Persistenz:** ja, pro Nutzer, fallübergreifend.

## Architektur

```
Frontend-Tab „Wissensbasis"
  ├─ Suche  → POST /v1/rag/retrieve   (bestehender Proxy + neues collection-Feld)
  └─ Chat   → POST /v1/rag/ask/stream (neu; retrieve → Prompt → Modell-Streaming)
                └─ rag_chats (neu, JSONB-Messages inkl. Quellen)
debian RAG-API (Hybrid-Retrieval, RRF, optional Reranker; Wake-über-SSH via rag_context)
```

## Backend

### Suche

- `RagRetrieveRequest` (shared.py) bekommt `collection: Optional[str]`,
  Allowlist `{kanzlei, jurisprudence, doktrin}`; Proxy reicht das Feld durch
  (debian-API unterstützt es bereits, `rag_context` sendet es heute schon).
- „Alle" löst das Frontend durch drei parallele Retrieves; Anzeige gruppiert
  nach Collection. Kein Misch-Ranking (RRF-Scores sind über Collections nicht
  vergleichbar). Reranker an, Limit 8 je Collection.

### Chat: `POST /v1/rag/ask/stream` (neuer Router-Teil in `endpoints/rag.py`)

Request: `{question, collections: [..], model, chat_id?: uuid}`.

1. Retrieval über `rag_context.retrieve_chunks` je gewählter Collection
   (damit greift die Wake-debian-Logik). **Fail-closed:** 0 Chunks insgesamt →
   Fehlermeldung statt Antwort, es wird nie aus reinem Modellwissen geantwortet.
2. Prompt: Chunks nummeriert `[Q1]…[Qn]`, je Chunk eine Herkunftszeile
   (Collection + `context_header`). Verlauf (letzte ~12 Messages aus dem Chat)
   wie im Befragen-Chat. System-Prompt-Kern:
   - jede tatsächliche/rechtliche Aussage mit `[Qn]` belegen,
   - wörtliche Übernahmen in Anführungszeichen mit `[Qn]`,
   - was die Quellen nicht tragen, explizit als „im Bestand nicht belegt"
     kennzeichnen — niemals frei ergänzen,
   - keine Semikola (Kanzlei-Stilregel).
3. Modell-Routing wie in `query.py` (gpt* → OpenAI, claude* → Anthropic,
   grok* → xAI, sonst Gemini) — nur Text, keine Datei-Uploads nötig.
4. Nach Stream-Ende: Frage + Antwort + verwendete Quellen-Chunks + Modell
   serverseitig an den Chat anhängen (Titel = erste Frage, gekürzt).
5. Deterministischer Post-Check: Antwort ohne `[Qn]`-Marker → Flag
   `unbelegt: true` am persistierten Message-Objekt; Frontend zeigt Warnbanner.

### Chat-Persistenz

- Tabelle `rag_chats`: `id UUID PK`, `owner_id FK users`, `title`,
  `collections JSONB`, `messages JSONB` (`[{role, content, model,
  sources: [chunk…], unbelegt?, created_at}]`), `created_at`, `updated_at`.
- ORM-Modell in `models.py` + neuer Eintrag in `MIGRATIONS` (main.py,
  idempotent, kein Alembic).
- Endpoints (`endpoints/rag.py`): `GET /v1/rag/chats` (Liste: id, title,
  updated_at), `GET /v1/rag/chats/{id}` (voll), `DELETE /v1/rag/chats/{id}`.
  Chat entsteht implizit beim ersten `ask` ohne `chat_id`. Alles per
  `owner_id` gescopet, kein `case_id`.

## Frontend (`index.html` + `app.js`)

- Dritter Haupt-Tab „📚 Wissensbasis" in der bestehenden Tab-Leiste.
- Kopfzeile: Eingabefeld, Collection-Dropdown (Alle/Kanzlei/Rechtsprechung/
  Doktrin), Modell-Dropdown, Buttons „Suchen" und „Fragen".
- Suchergebnis: Karten je Chunk — Score, Herkunft (`context_header`),
  Text, Metadata-Badges (Schlagwörter/Normen, sofern vorhanden), gruppiert
  nach Collection. Reine Anzeige, keine Filter-UI (v1).
- Chat: Antwort als Markdown, `[Qn]`-Marker anklickbar → scrollt/klappt die
  Quelle unter der Antwort auf („Verwendete Quellen"). Warnbanner bei
  `unbelegt`. Verlauf-Liste (wie Entwurf-Verlauf): laden, fortsetzen, löschen.

## Fehlerbehandlung

- RAG nicht erreichbar → deutsche Meldung „Wissensbasis (debian) wird
  geweckt — bitte in einer Minute erneut versuchen" (Wake läuft automatisch an).
- 0 Treffer beim Fragen → „Keine Belege im Bestand gefunden" statt Antwort.
- LLM-Fehler wie im Befragen-Feld (Fehlertext im Stream).

## Tests

Pytest mit gestubbtem httpx/LLM-Clients: Collection-Allowlist des Proxys,
Prompt-Assembly (Nummerierung, Herkunftszeilen, Verlaufs-Trimmung),
Fail-closed bei leerem Retrieval, `[Qn]`-Post-Check, Chat-CRUD + Scoping
(fremder Nutzer → 404), Migrations-Idempotenz.

## Bewusst weggelassen (v1)

Facetten-Filter-UI, Fall-Scoping der Chats, Upsert-UI, Misch-Ranking über
Collections, Chat-Sharing zwischen Nutzern.

## Backlog danach (separat, nicht Teil dieser Spec)

1. `verify-facts`/`draft-context` als Buttons am Entwurf.
2. Citation-Verifier-Ergebnisse am Draft anzeigen.
3. Jurisprudence-Packs am Fall sichtbar machen.
