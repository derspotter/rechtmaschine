# Flair Anonymization Service - Debug Status

**Date:** 2026-01-15
**Current Version:** v3.0.3
**Status:** ✅ RESOLVED - PLZ_CITY_PATTERN regex was matching across newlines

---

## Resolution (2026-01-15)

**Root Cause:** The `PLZ_CITY_PATTERN` regex was using `\s+` which matches ALL whitespace characters including newlines. This caused it to incorrectly match patterns like "35880\nEs" (translator ID followed by newline and capitalized word).

**Example of the bug:**
```python
# OLD regex (buggy):
PLZ_CITY_PATTERN = re.compile(
    r'(?:,\s*)?(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
)
# Matched: "35880\nEs" → thought it was postal code + city!
```

**Fix Applied:**
```python
# NEW regex (fixed):
PLZ_CITY_PATTERN = re.compile(
    r'(?:,[ \t]*)?(\d{5})[ \t]+([A-ZÄÖÜ][a-zäöüß]+(?:[ \t]+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
)
# Now only matches spaces/tabs, NOT newlines
```

**Testing Results:**
- ✅ "Nr. 35880" → kept unchanged (no longer replaced as [ORT])
- ✅ "53115 Bonn" → correctly anonymized as [ORT]
- ✅ Phone numbers no longer incorrectly tagged

**Commits:**
- v3.0.3: Fixed PLZ_CITY_PATTERN to not match across newlines
- Removed debug logging (no longer needed)

---

## Original Issue (For Reference)

The simplified Flair anonymization service (`anonymization_service_flair_simple.py`) is replacing numbers with `[ORT]` when they shouldn't be:

**Example:**
```
Original: Als Sprachmittler ist anwesend: Frau/Herr Nr. 35880
Anonymized: Als Sprachmittler ist anwesend: Frau/Herr Nr. [ORT]
```

The translator ID "35880" is being tagged as ST (Stadt/City) by Flair and replaced with `[ORT]`.

---

## What We Implemented

### Numeric Filter (Line 185-191)
```python
elif ent['tag'] == 'ST':  # Stadt (City)
    # Filter out numeric-only detections (false positives like translator IDs)
    entity_text = ent['text'].strip()
    is_numeric = entity_text.isdigit()
    print(f"[DEBUG] ST tag: '{ent['text']}' (stripped: '{entity_text}', isdigit: {is_numeric})")
    if not is_numeric:
        entities_to_replace.append((ent['start'], ent['end'], '[ORT]'))
```

**Goal:** Skip ST tags that are purely numeric (like "35880")

---

## Test Setup

### 1. OCR Text File
We have extracted text saved in `/tmp/ocr_text.txt` (8211 chars from a BAMF Anhörung).

**Key lines to check:**
- Line 5: `Referat 42H AS Im AZ Bonn` → Should become `... [ORT]` ✓
- Line 9: `53115 Bonn` → Should become `[ORT]` ✓
- Line 27: `Als Sprachmittler ist anwesend: Frau/Herr Nr. 35880` → Should keep "Nr. 35880" ✗ (Currently becomes `Nr. [ORT]`)

### 2. Test Command
```bash
python3 << 'EOF'
import requests

with open('/tmp/ocr_text.txt', 'r') as f:
    text = f.read()

resp = requests.post('http://localhost:9002/anonymize',
    json={'text': text, 'document_type': 'Anhörung'},
    timeout=60)

result = resp.json()
print('Status:', resp.status_code)
print('Version:', requests.get('http://localhost:9002/health').json().get('version'))
print()

# Check the problematic line
lines = result['anonymized_text'].split('\n')
for i, line in enumerate(lines[:30]):
    if 'Sprachmittler' in line or 'Nr.' in line:
        print(f'Line {i}: {line}')
EOF
```

### 3. Expected vs Actual

**Expected output for line 27:**
```
Als Sprachmittler ist anwesend: Frau/Herr Nr. 35880 wird durchgeführt von: Frau / Herr [PERSON]
```

**Actual output:**
```
Als Sprachmittler ist anwesend: Frau/Herr Nr. [ORT] wird durchgeführt von: Frau / Herr [PERSON]
```

---

## Debug Steps

### Step 1: Check Debug Logs
The service should print `[DEBUG] ST tag:` lines when processing. Look for:
```
[DEBUG] ST tag: '35880' (stripped: '35880', isdigit: True)
```

If this appears, the filter is working but being overridden later.
If this DOESN'T appear, Flair isn't tagging "35880" as ST at all - something else is.

### Step 2: Check Regex Patterns
The PLZ_CITY_PATTERN might be matching incorrectly:
```python
PLZ_CITY_PATTERN = re.compile(
    r'(?:,\s*)?(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
)
```

Test it:
```python
import re
PLZ_CITY_PATTERN = re.compile(
    r'(?:,\s*)?(\d{5})\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\b'
)

test = "Als Sprachmittler ist anwesend: Frau/Herr Nr. 35880"
matches = list(PLZ_CITY_PATTERN.finditer(test))
print(f"PLZ pattern matches: {len(matches)}")
for m in matches:
    print(f"  Matched: '{m.group(0)}'")
```

**Expected:** 0 matches (regex should NOT match "Nr. 35880")

### Step 3: Check All Regex Patterns
Other regex patterns that could match:
- `DOB_PATTERN` - dates (shouldn't match "35880")
- `ADDRESS_PATTERN` - streets with numbers (shouldn't match)
- `ADDRESS_CUE_PATTERN` - address with context (shouldn't match)

---

## Possible Root Causes

### Theory 1: Regex is matching
The PLZ_CITY_PATTERN or another regex is incorrectly matching "Nr. 35880".

**Test:** Disable ALL regex (comment out lines 195-231) and see if issue persists.

### Theory 2: Flair is tagging it as ST
Flair's NER model incorrectly tags "35880" as ST (Stadt).

**Test:** Look for `[DEBUG] ST tag: '35880'` in logs. If present, the numeric filter should catch it.

### Theory 3: Text offset issue
The entity offsets are wrong, causing the replacement to hit the wrong position.

**Test:** Add more debug logging to show entity positions:
```python
print(f"[DEBUG] ST at pos {ent['start']}-{ent['end']}: '{ent['text']}'")
```

### Theory 4: Overlapping entity deduplication
The deduplication logic (lines 237-251) might be choosing the wrong entity to keep.

**Test:** Log all `entities_to_replace` before deduplication:
```python
print(f"[DEBUG] Total entities before dedup: {len(entities_to_replace)}")
for start, end, repl in entities_to_replace[:20]:  # First 20
    print(f"  {start}-{end}: '{text[start:end]}' → {repl}")
```

---

## Quick Fixes to Try

### Fix 1: More Aggressive Numeric Filter
Instead of just checking if the ST tag is all digits, also check if it's a short number:
```python
elif ent['tag'] == 'ST':
    entity_text = ent['text'].strip()
    # Skip if purely numeric OR if it's a short number (likely an ID)
    if entity_text.isdigit() or (entity_text.isdigit() and len(entity_text) <= 6):
        continue
    entities_to_replace.append((ent['start'], ent['end'], '[ORT]'))
```

### Fix 2: Context-Based Filter
Check the surrounding text for clues:
```python
elif ent['tag'] == 'ST':
    entity_text = ent['text'].strip()
    # Check 20 chars before entity for "Nr." or "ID"
    context_before = text[max(0, ent['start']-20):ent['start']].lower()
    is_id = 'nr.' in context_before or 'id' in context_before or 'az' in context_before

    if entity_text.isdigit() or is_id:
        print(f"[DEBUG] Skipping ST tag (numeric or ID context): '{entity_text}'")
        continue
    entities_to_replace.append((ent['start'], ent['end'], '[ORT]'))
```

### Fix 3: Disable ST Tag Entirely
If ST tags are too unreliable, just don't anonymize cities:
```python
elif ent['tag'] == 'ST':
    pass  # Don't anonymize cities - too many false positives
```

This is GDPR-safe (cities aren't personal data) but less thorough.

---

## Service Commands

### Start Service
```bash
cd /var/opt/docker/rechtmaschine/anon
python anonymization_service_flair_simple.py
```

### Check Health
```bash
curl http://localhost:9002/health | jq .
```

### Test Anonymization
```bash
curl -X POST http://localhost:9002/anonymize \
  -H "Content-Type: application/json" \
  -d @- << 'EOF' | jq -r '.anonymized_text' | head -30
{
  "text": "Als Sprachmittler ist anwesend: Frau/Herr Nr. 35880\nEs erscheint Herr/Frau Goudarzi, Hadi geb. 22.03.1997.",
  "document_type": "Anhörung"
}
EOF
```

---

## Files

- **Service:** `/var/opt/docker/rechtmaschine/anon/anonymization_service_flair_simple.py`
- **Test data:** `/tmp/ocr_text.txt` (8211 chars from real Anhörung)
- **This doc:** `/var/opt/docker/rechtmaschine/anon/ANONYMIZATION_DEBUG.md`

---

## Architecture Decision

**Context:** We switched from Qwen 14B LLM to Flair NER for speed, but Flair can't understand context.

**Trade-off:**
- **Qwen 14B:** Slow (~10-15s), understands context, accurate categorization
- **Flair NER:** Fast (~1-3s), no context understanding, false positives

**Current approach:** Simple mode - anonymize ALL people regardless of role.
**GDPR rationale:** Over-anonymization is safer than under-anonymization.

**If false positives persist:** Consider hybrid approach (Codex recommended):
1. Flair NER finds all person mentions (fast)
2. Small 7B LLM categorizes each with 2-3 sentence context (faster than 14B)
3. Aggregate roles document-wide

---

## Next Steps

1. ✅ Add debug logging for ST tags (committed in e2d8ea7)
2. ⏳ Run test and check logs for `[DEBUG] ST tag:` output
3. ⏳ Identify which pattern (Flair ST or regex) is causing "[ORT]"
4. ⏳ Apply appropriate fix (numeric filter, context filter, or disable ST)
5. ⏳ Test with full Anhörung document
6. ⏳ Remove debug logging once fixed
7. ⏳ Update main app to use simplified service

---

## Contact

Issues or questions: This was debugged with Claude Sonnet 4.5 on 2026-01-14.
Latest commit: e2d8ea7 (debug logging for ST tags)
