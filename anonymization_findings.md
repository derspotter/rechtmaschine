# Anonymization Findings (2026-01-19)

## Scope
- Sample document: `test_files/ocred/ocr_bescheid_clean.md`
- Goal: reliable anonymization of plaintiff names + DOBs + addresses (phones optional)
- IBAN/BIC anonymization is not required

## Current Pipeline State
- App sends first 5k chars to LLM; Qwen service also capped at 5k chars.
- Regex is applied to the remaining text using extracted names, birth_dates, and addresses.
- `service_manager.py` can switch anonymization backend via `ANON_BACKEND=flair|qwen`.

## Qwen LLM Prompt (qwen3:14b)
- Extracts and anonymizes: names, DOBs (Geburtsdatum/geb.), addresses.
- Outputs JSON with `plaintiff_names`, `birth_dates`, `addresses`, `anonymized_text`.
- Still limited by 5k window and exact-match regex on tail.

## Flair + Regex (current SOTA)
Changes applied:
- DOB redaction via cue-based regex + DOB blocks.
- Address regex expanded to allow hyphens and strasse/str.
- Aktenzeichen cues expanded; AZR + BAMF numeric IDs near cues flagged.

Latest result on `ocr_bescheid_clean.md`:
- Misses: names 0, dobs 0, addresses 0
- Remaining: phones 1, IDs 5 (AZR/BAMF numbers)
- Runtime ~29s

## LLM + Regex (qwen3:14b)
Latest result on `ocr_bescheid_clean.md`:
- Misses: names 3 (variant forms), dobs 2, addresses 10, IDs 9
- Runtime ~88s
- Root causes: limited 5k window + exact string matching on name variants

## Outstanding Problems
- AZR/BAMF IDs still not fully redacted (multi-line numbers, repeated headers).
- Phone numbers still present (not a priority but still leaking PII if needed).
- LLM approach is slower and misses variant names/addresses beyond the 5k window.

## Recommendations
- Keep Flair + regex as the primary pipeline.
- Add stronger AZR/BAMF ID redaction across lines/headers if required.
- Treat phone redaction as optional (current leakage is limited to header phone).

## Next Steps
- Extend AZR/BAMF ID redaction to catch long numeric sequences near labels across line breaks.
- Consider optional phone regex if required by policy.
- Add name-variant normalization (surname-first, OCR spacing) if we revisit LLM extraction.

## How to Run Tests
### 1) NER + regex evaluation
```bash
/home/jayjag/rechtmaschine/anon/.venv/bin/python - <<'PY'
import os
import re
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
sys.path.append("/home/jayjag/rechtmaschine/anon")
from anonymization_service_flair import anonymize_with_flair

DOC_PATH = Path("/home/jayjag/rechtmaschine/test_files/ocred/ocr_bescheid_clean.md")
text = DOC_PATH.read_text(encoding="utf-8")

LEGAL_TOKENS = [
    "egmr",
    "urteil",
    "entscheidung",
    "beschluss",
    "bverwg",
    "vgh",
    "vg ",
    "vg,",
    "rn.",
    "abs.",
    "asyl",
    "dublin",
    "grch",
    "vo",
]


def has_legal_token(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in LEGAL_TOKENS)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_expected(text: str) -> dict:
    lines = [line.rstrip() for line in text.splitlines()]
    trimmed = [line.strip() for line in lines if line.strip()]

    names = set()
    name_trigger = re.compile(r"\bName\b", re.IGNORECASE)
    name_block = False

    for line in trimmed:
        if name_trigger.search(line) or "Vorname/NAME" in line:
            name_block = True
            continue
        if name_block and re.search(r"Geburtsdatum|Aktenzeichen|Anlagen", line, re.IGNORECASE):
            name_block = False

        if name_block:
            candidate = line.strip()
            if re.match(r"^\d{2}\.\d{2}\.\d{4}$", candidate):
                continue
            if re.search(r"\d", candidate):
                continue
            if has_legal_token(candidate):
                continue
            if re.match(
                r"^[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}\s+[A-ZÄÖÜ]{2,}$",
                candidate,
            ):
                names.add(candidate)
            if re.match(
                r"^[A-ZÄÖÜ]{2,},\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2}$",
                candidate,
            ):
                names.add(candidate)

    for line in trimmed:
        if not re.search(r"geb\.?\s*(am)?\s*\d{2}\.\d{2}\.\d{4}", line, re.IGNORECASE):
            continue
        for match in re.finditer(
            r"\b([A-ZÄÖÜ]{2,},\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){0,2})",
            line,
        ):
            candidate = match.group(1)
            if not has_legal_token(candidate):
                names.add(candidate)

    dobs = set()
    for line in trimmed:
        if re.search(r"\bgeb\.?\s*(am)?\b", line, re.IGNORECASE):
            dobs.update(re.findall(r"\b\d{2}\.\d{2}\.\d{4}\b", line))

    for i, line in enumerate(trimmed):
        if re.search(r"Geburtsdatum", line, re.IGNORECASE):
            for j in range(i + 1, min(i + 12, len(trimmed))):
                candidate = trimmed[j].strip()
                if not candidate:
                    break
                if re.search(r"\bName\b|Aktenzeichen|Anlagen", candidate, re.IGNORECASE):
                    break
                if re.match(r"^\d{2}\.\d{2}\.\d{4}$", candidate):
                    dobs.add(candidate)

    addresses = set()
    street_regex = re.compile(
        r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+)*\s*(?:str\.|straße|strasse|allee|weg|platz|ring|gasse|damm|ufer)\s*\d+[a-zA-Z]?\b",
        re.IGNORECASE,
    )
    street_compact = re.compile(
        r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß\-]+str\.?\s*\d+[a-zA-Z]?\b",
        re.IGNORECASE,
    )
    postcode_regex = re.compile(
        r"\b\d{5}\s*[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)*\b"
    )

    for line in trimmed:
        if has_legal_token(line):
            continue
        if ";" in line:
            continue
        if len(line) > 80:
            continue
        for match in street_regex.finditer(line):
            addresses.add(match.group(0))
        for match in street_compact.finditer(line):
            addresses.add(match.group(0))
        for match in postcode_regex.finditer(line):
            addresses.add(match.group(0))

    phones = set()
    for line in trimmed:
        if re.search(r"\bTel", line, re.IGNORECASE) or re.search(r"\+\d", line):
            for match in re.finditer(r"\+?\d[\d\s/.-]{6,}\d", line):
                phones.add(match.group(0).strip())

    ids = set()
    for line in trimmed:
        for match in re.findall(r"\b\d{6,}-\d{1,}\b", line):
            ids.add(match)
        if "AZR" in line:
            ids.update(re.findall(r"\b\d{9,}\b", line))

    return {
        "names": sorted(names),
        "dobs": sorted(dobs),
        "addresses": sorted(addresses),
        "phones": sorted(phones),
        "ids": sorted(ids),
    }


def leak_details(expected: dict, anonymized_text: str) -> dict:
    normalized = normalize(anonymized_text)
    leaks = {}
    for key, values in expected.items():
        leaked = []
        for item in values:
            if normalize(item) in normalized:
                leaked.append(item)
        leaks[key] = leaked
    return leaks


expected = extract_expected(text)
print("Expected counts:", {k: len(v) for k, v in expected.items()})

start = time.time()
flair_anonymized, _, _, _, _ = anonymize_with_flair(text)
flair_elapsed = time.time() - start
flair_leaks = leak_details(expected, flair_anonymized)

print(f"\nNER+regex (Flair) time: {flair_elapsed:.1f}s")
print("Missed items (still present):")
for key in ["names", "dobs", "addresses", "phones", "ids"]:
    values = flair_leaks.get(key, [])
    print(f"- {key}: {len(values)}")
    for value in values:
        print(f"  * {value}")
PY
```

### 2) Run services locally
```bash
# Start service manager (Flair backend)
ANON_BACKEND=flair python service_manager.py

# Start service manager (Qwen backend)
ANON_BACKEND=qwen python service_manager.py
```

### 3) Call anonymize endpoint
```bash
# Replace FILE.pdf and DOCUMENT_TYPE as needed
curl -s -X POST http://localhost:8004/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "<paste text>", "document_type": "Bescheid"}'
```
