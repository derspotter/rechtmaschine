# Anonymization Tightening — Desktop Service Improvements

Based on a 55-file bench run against `anonymization_service.py` (Qwen3:8b hybrid LLM+regex).
26 of 55 files had real name leaks (109 instances total).

---

## 1. MRZ Line Stripping (Critical — Low Effort)

ID document scans contain machine-readable zone lines like:
```
NAD<<SANAIPOUR<<HOSSEIN<<<<<<<<<<<<<
```
These are never caught by `safe_replace` because the name is encoded with `<<` delimiters.

**Fix:** Add a regex safety catch in `apply_regex_replacements`:
```python
# Safety catch: MRZ lines from ID document scans
anon = re.sub(r"[A-Z]{1,3}<<[A-Z]+<<[A-Z]+<*", "[MRZ-REDACTED]", anon)
```

---

## 2. OCR Confusable Characters (Critical — Medium Effort)

OCR frequently substitutes visually similar characters:
- `GOUDARZl` (lowercase L) instead of `GOUDARZI` (uppercase I)
- `Sanalpour` instead of `Sanaipour` (l/i swap)
- `Daniel-Gofdbach-StraBe` instead of `Goldbach-Straße` (B/ß, fd/ld)

`_escape_fuzzy` only handles spacing. It doesn't handle character-level OCR drift.

**Fix:** Add an OCR confusable map to `_escape_fuzzy`:
```python
OCR_CONFUSABLES = {
    "I": "[Il1\|]",
    "l": "[Il1\|]",
    "1": "[Il1]",
    "O": "[O0]",
    "0": "[O0]",
    "ß": "[ßBs]",
    "B": "[Bß8]",
    "rn": "(?:rn|m)",
    "m": "(?:m|rn)",
}
```
After `re.escape`, walk the pattern and expand each character through the map. Bounded set — won't cause false positives on legal text.

---

## 3. Individual Name Token Matching (High — Low Effort)

LLM extracts `"Pourya Goudarzi"` but text has `"Sohn: Pourya GOUDARZl"` — the compound doesn't match. Even if compound fails, individual tokens should be tried.

`_person_term_variants` only generates comma-flipped variants. It doesn't split names into tokens.

**Fix:** In `_person_term_variants`, also yield each token ≥4 characters:
```python
# After existing variant generation:
tokens = [t for t in clean.split() if len(t) >= 4 and t.lower() not in HONORIFICS]
for token in tokens:
    if token.lower() not in seen:
        seen.add(token.lower())
        out.append(token)
```
Short tokens (≤3 chars like "Ali") risk false positives — skip them or require `\b` word boundaries.

---

## 4. Bescheid Header Pattern (High — Low Effort)

Bescheid preambles have a structural pattern:
```
In dem Asylverfahren der
1. ELBIF, Mona, geb. am ...
2. ELBIF, Khadija, geb. am ...
```

LLM may return normalized `"Mona Elbif"` but text has `"ELBIF, Mona"`. Even with comma flipping, OCR garble can break matching.

**Fix:** Add a structural regex safety catch:
```python
# Safety catch: numbered plaintiff lists in Bescheid headers
anon = re.sub(
    r"(\d+\.\s+)([A-ZÄÖÜ][A-ZÄÖÜa-zäöüé\-]+),\s+([A-ZÄÖÜa-zäöüé\-]+(?:\s+[A-ZÄÖÜa-zäöüé\-]+)*)",
    r"\1[PERSON]",
    anon,
)
```
This is deterministic and doesn't depend on LLM extraction accuracy.

---

## 5. Second-Pass Verification (High — Low Effort)

Pipeline is: LLM extract → regex replace → done. No verification.

**Fix:** After `apply_regex_replacements`, loop over all extracted names (including individual tokens) and check if any still appear in the text. If found, force-replace and log a warning:
```python
# Verification pass
all_names = entities.get("names", [])
for name in all_names:
    for token in name.split():
        if len(token) >= 4 and token.lower() in anonymized_text.lower():
            print(f"[WARN] Name token '{token}' survived replacement, force-replacing")
            anonymized_text = re.sub(re.escape(token), "[PERSON]", anonymized_text, flags=re.IGNORECASE)
```

---

## 6. BAMF Officer / Lawyer Names (Medium — Low Effort)

Signing officials (Varga, Witta, Reinhardt) and lawyers (Keienborg) aren't anonymized. The LLM sometimes excludes them.

**Fix (structural regex):** Catch signature blocks without relying on LLM:
```python
# Safety catch: official signatures
anon = re.sub(r"(?:gez\.|gezeichnet)\s+([A-ZÄÖÜ][a-zäöü]+)", r"gez. [BEAMTER]", anon)
anon = re.sub(r"Einzelentscheider(?:/in)?:\s*([A-ZÄÖÜ][a-zäöü]+)", r"Einzelentscheider/in: [BEAMTER]", anon)
```

---

## 7. ALL CAPS Surface Form Variants (Medium — Low Effort)

LLM returns `Goudarzi` (title case) but text has `GOUDARZI` (all caps). `re.IGNORECASE` handles straight matching, but OCR confusable logic needs the actual surface form.

**Fix:** Auto-generate an uppercase variant for every extracted name:
```python
# In safe_replace, before variant loop:
if term != term.upper():
    sorted_terms.append(term.upper())
```

---

## Priority Summary

| # | Issue | Severity | Effort | Leaks Fixed |
|---|-------|----------|--------|-------------|
| 1 | MRZ line stripping | Critical | Low | SANAIPOUR MRZ |
| 2 | OCR confusable chars | Critical | Medium | GOUDARZl, Sanalpour |
| 3 | Individual token matching | High | Low | Pourya, Ziba, Zivdar |
| 4 | Bescheid header pattern | High | Low | ELBIF family preamble |
| 5 | Second-pass verification | High | Low | All residual leaks |
| 6 | Official/lawyer names | Medium | Low | Varga, Witta, Keienborg |
| 7 | ALL CAPS variants | Medium | Low | GOUDARZI, SANAIPOUR |

Items 1-5 would eliminate all leaks found in the bench run. Items 6-7 are polish.
