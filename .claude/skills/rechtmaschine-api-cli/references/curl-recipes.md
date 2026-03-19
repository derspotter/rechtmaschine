# Curl Recipes

Assume:

```bash
BASE_URL="https://rechtmaschine.de"
EMAIL="der_spotter"
PASSWORD="..."
```

For local testing:

```bash
BASE_URL="http://127.0.0.1:8000"
```

## 1. Login and store token

```bash
TOKEN=$(curl -s -X POST "$BASE_URL/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASSWORD" | jq -r '.access_token')
```

## 2. List cases

```bash
curl -s "$BASE_URL/cases" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## 3. List documents

```bash
curl -s "$BASE_URL/documents" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## 4. List saved sources

```bash
curl -s "$BASE_URL/sources" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## 5. Query documents

Use filenames for document selections and UUIDs for `saved_sources`.

```bash
curl -N -X POST "$BASE_URL/query-documents" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Was steht im Bescheid?",
    "model": "gemini-3-flash-preview",
    "chat_history": [],
    "selected_documents": {
      "anhoerung": [],
      "rechtsprechung": [],
      "saved_sources": [],
      "sonstiges": [],
      "akte": [],
      "bescheid": {
        "primary": "BESCHEID_DATEINAME.pdf",
        "others": []
      },
      "vorinstanz": {
        "primary": null,
        "others": []
      }
    }
  }'
```

## 6. Generate a draft

This endpoint streams NDJSON. Use `curl -N`.

```bash
curl -N -X POST "$BASE_URL/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "document_type": "Klagebegründung",
    "user_prompt": "Bitte schreibe eine Klagebegründung.",
    "model": "gpt-5.4",
    "verbosity": "medium",
    "chat_history": [],
    "selected_documents": {
      "anhoerung": ["ANHOERUNG.pdf"],
      "rechtsprechung": [],
      "saved_sources": ["UUID-DER-SOURCE"],
      "sonstiges": [],
      "akte": [],
      "bescheid": {
        "primary": "BESCHEID.pdf",
        "others": []
      },
      "vorinstanz": {
        "primary": null,
        "others": []
      }
    }
  }'
```

Common `document_type` values:
- `Klagebegründung`
- `Antrag auf Zulassung der Berufung (AZB)`
- `Schriftsatz`

Common `model` values:
- `claude-opus-4-6`
- `gpt-5.4`
- `gemini-3.1-pro-preview`
- `two-step-expert`
- `multi-step-expert`

## 7. Run research

```bash
curl -s -X POST "$BASE_URL/research" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Aktuelle Rechtsprechung zu Afghanistan und Sippenhaft",
    "search_engine": "meta",
    "search_mode": "balanced",
    "max_sources": 12,
    "selected_documents": {
      "anhoerung": [],
      "rechtsprechung": [],
      "saved_sources": [],
      "sonstiges": [],
      "akte": [],
      "bescheid": {
        "primary": "BESCHEID.pdf",
        "others": []
      },
      "vorinstanz": {
        "primary": null,
        "others": []
      }
    }
  }' | jq
```

## 8. Read research history

```bash
curl -s "$BASE_URL/research/history?limit=20" \
  -H "Authorization: Bearer $TOKEN" | jq
```

## Troubleshooting

## 401 Unauthorized
- Token missing or expired
- Log in again via `/token`

## 422 Validation error
- Wrong `document_type`
- Used titles instead of filenames
- Used titles instead of UUIDs in `saved_sources`

## Source is visible in UI but not uploadable in generation
- `saved_source` may be link-only
- Check `download_path` / `download_status`
- Link-only sources can still be embedded as inline text in generation

## Query works without JPEG but fails with JPEG
- Gemini query path should now accept JPEG/PNG as image parts
- If it fails again, inspect `app/endpoints/query.py` and app logs
