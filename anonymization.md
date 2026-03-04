# Anonymization: Problem History, Experiments, and Current State

Date: 2026-03-01
Scope: German asylum document anonymization (Bescheid / Anhoerung and related attachments)

## 1. Executive Summary

The core problem was not one single bug. It was a combination of:

1. Gemma3 structured-output instability in Ollama (hangs/very slow runs with schema-constrained JSON).
2. Recall gaps on hard OCR/text-noisy name/address/id patterns.
3. Throughput pressure from multi-step extraction + external verification.

We iterated through model settings, prompting, architecture changes, and deterministic post-filters.
The strongest reliability gains came from:

1. Gemma3 using `format: "json"` (not schema dict) for extraction stability.
2. Staged extraction (names, addresses, birth/ids) with higher effort on names.
3. Aggressive deterministic regex/context catches for known leak classes.
4. Continuous benchmark harness with Claude verification.

Current status at time of writing:

- Hard fail cohort (`input_fail10_rerun_20260226-163249`) reached `10/10 PASS` in run `staged-split-fail10-r12-20260301-142548`.
- A broader 50-file run was started (`staged-split-noocr50-r1-20260301-144602`) but intentionally stopped by user; partial state was `6 processed, 3 PASS / 3 FAIL`.
- Remaining partial-run fails were clustered, not random:
  - OCR artifact in person field (`Bpchum` in Bearbeiter context)
  - Private address variant (`Lohmaer Weg 4, 07907 Schleiz`)

## 2. System Context

### 2.1 Runtime architecture

Anonymization is done via server endpoint `app/endpoints/anonymization.py`.
It calls desktop/service-manager for extraction and applies replacement logic locally.

High-level pipeline:

1. Load text (OCR may have happened earlier depending on source path).
2. Extract entities with LLM via `/extract-entities`.
3. Filter/augment entities.
4. Apply deterministic regex replacements over original text.
5. Return anonymized text + metadata.

### 2.2 Verification architecture

`tests/pipeline_anonymize_claude_verify.sh` runs document batches and verifies each output with Claude CLI.
Important operational detail:

- Claude is run keyless with Max session semantics:
  - `env -u ANTHROPIC_API_KEY -u CLAUDE_API_KEY claude -p`

## 3. Primary Failure Modes We Observed

## 3.1 Gemma3 structured output hang / slowdown

Documented in `docs/bug-gemma3-structured-output-hang.md`.
Observed behavior:

- Some repetitive documents intermittently took hundreds of seconds or timed out.
- Behavior was non-deterministic (same payload could be fast or hang).

Contributing factors we treated as likely causes:

1. Low-temp Gemma3 behavior regressions in Ollama versions.
2. Structured-output schema/grammar overhead.
3. Repetitive text causing pathological sampling loops.

## 3.2 Extraction misses on noisy identity fields

Typical misses:

- Signature OCR garbage that still looks person-like.
- Staff/signer names in narrow layout blocks.
- Address fragments in homeland-address narrative form (no house number pattern).
- Internal ID patterns outside standard labels.
- Certificate signer/name fragments where OCR and placeholder boundaries mix.

## 3.3 Verification false positives

Claude occasionally flagged legal references or metadata that are not PII leaks.
We tuned evaluation instructions, but verifier interpretation still needs strict, repeated calibration.

## 4. Strategies We Tried (Chronological / Thematic)

## 4.1 Structured output strategy: schema dict -> `format: "json"` for Gemma3

Change:

- For `gemma3:*`, extraction uses `format: "json"` instead of schema dict.

Why:

- Reduce grammar-constrained decoding pathologies while still enforcing JSON-only output shape via prompt.

Result:

- Significant stability improvement vs schema-constrained runs.
- Did not fully solve recall; deterministic post-filters still required.

## 4.2 Prompt strategy iterations

We iterated prompts in multiple directions:

1. Shorter, stricter extraction instructions.
2. Explicit "do not treat tribes/ethnicities/religions/nationalities as names".
3. Added role/signature contexts (for example "Im Auftrag", signer blocks).
4. Added explicit family-relation extraction cues (Vater/Mutter/etc).
5. Added stricter "extract each person once" guidance.

Result:

- Prompt changes helped, but never solved long-tail leaks alone.
- Best outcomes always came from prompt + deterministic post-processing.

## 4.3 Sampling strategy (temperature)

We tested multiple Gemma temperatures (including repeated runs).
Operationally, 0.5 was frequently used for names-stage robustness.

Observed tradeoff:

- Lower temp can improve consistency but may trigger latency pathologies in some scenarios.
- Higher temp can improve recall in specific hard cases but adds variance.

Conclusion:

- Single global temperature was not ideal.
- Stage-specific temperatures performed better.

## 4.4 Context/timeout strategy

Tuned `num_ctx` and timeouts.

Findings:

- 90s API timeout still produced occasional skips on larger/harder files.
- 120s reduced timeout-driven non-passes in hard cohorts.
- Long context is not always better; too-long/too-noisy payloads can hurt reliability and latency.

## 4.5 Architecture strategy: staged extraction

Implemented staged extraction in `app/endpoints/anonymization.py`:

1. names
2. addresses
3. birth_ids

Current stage-specific behavior:

- Names: typically 2 passes for Gemma3 (`OLLAMA_NAMES_PASSES_GEMMA3`, default 2)
- Addresses: 1 pass
- Birth/IDs: 1 pass
- Stage temperatures are independent env-driven controls.

Why this helped:

- Names were the highest-risk recall class.
- Splitting tasks reduced prompt overload and made tuning per-entity-type possible.

Cost:

- 4 Gemma calls per doc (2 + 1 + 1), excluding Claude verification.

## 4.6 Regex and deterministic safety-net hardening

Most reliability gains ultimately came here.

Major classes added/refined in `anon/anonymization_service.py` include:

1. Signature context catches:
   - Near "Unterschrift" labels
   - Near signer role labels (Betroffene, Sprachmittler, Sachbearbeiter)
   - Name-like lines before page footer (`Seite X von Y`)

2. Certificate context catches:
   - Aviation certificate signer tokens (`CAP...`, `CAPT.` and uppercase line variants)
   - Partial placeholder-fragment cleanup in OCR signer tokens
   - Certificate number/date/expiry field sanitization

3. Birth/date/place context catches:
   - `Geburtsdatum/Geburtstag` follow-up lines
   - `im Jahr 1992 geborene...` style phrases
   - `Geburtsort` table follow-up value

4. Family relation catches:
   - `Vater:` / `Mutter:` patterns including `(verstorben)` style suffix

5. Address context catches:
   - Labeled private address blocks
   - Homeland-address narrative variants (street/gasse/boulevard/siedlung/viertel style)
   - PLZ + city normalization patterns

6. ID/document number catches:
   - Interne Nummer / Aktenvorblatt style IDs
   - Additional ID morphology variants

7. Staff-role context catches:
   - `Bearbeiter`, `Sachbearbeiter`, `Geprueft`, `Im Auftrag`, `Verfuegung` neighbor lines

This was the decisive layer for long-tail leak suppression.

## 4.7 Alternative model strategies tested

We also explored alternatives and fallbacks (including desktop-side tests):

1. Qwen + Flair hybrid route (kept available as engine fallback).
2. Larger Qwen variants.
3. Comparative checks against Gemma12b behavior on problematic docs.

Outcome:

- Gemma path with hardened deterministic rules remained the mainline path.
- Multi-engine fallback remains useful but increases operational complexity.

## 4.8 Batch benchmarking strategy

We built/used continuous benchmark loops with:

- fixed cohorts
- rerun cohorts
- fail-only cohorts
- Claude verdict per file
- artifact capture (raw response, anonymized text, verdict json)

This made regressions visible and allowed targeted fixes instead of broad guesswork.

## 5. Why It Felt Slow

Short answer: each doc is not one call.

Current flow per doc (typical Gemma staged mode):

1. names pass 1
2. names pass 2
3. addresses pass
4. birth/ids pass
5. Claude verification pass (in benchmark mode)

So benchmark latency is dominated by:

- 4 LLM extraction calls per doc
- plus verifier call
- plus text processing and IO

If we optimize for speed only, we can reduce to names-only, 1 pass, and defer verification.

## 6. Representative Results

## 6.1 Hard fail cohort success

Run: `tmp/anonymize_pipeline/staged-split-fail10-r12-20260301-142548`

- Processed: 10
- Result: 10 PASS
- This cohort had previously exhibited both real leaks and timeout/skip behavior.

## 6.2 Broader 50-file run (stopped intentionally)

Run: `tmp/anonymize_pipeline/staged-split-noocr50-r1-20260301-144602`

State when stopped:

- Processed: 6
- Result: 3 PASS / 3 FAIL
- Fail clusters were narrow and actionable:
  - person-field OCR artifact class
  - specific private-address variant class

Interpretation:

- Not random catastrophic failure.
- Remaining issues are patchable deterministic classes.

## 7. Current Technical Knobs (Most Important)

Key controls in current code path:

1. `ANONYMIZATION_ENGINE_DEFAULT` (`gemma` or `qwen_flair`)
2. `OLLAMA_NUM_CTX_GEMMA3` and `OLLAMA_NUM_CTX_DEFAULT`
3. `OLLAMA_NAMES_TEMP_GEMMA3`
4. `OLLAMA_ADDRESSES_TEMP_GEMMA3`
5. `OLLAMA_BIRTH_IDS_TEMP_GEMMA3`
6. `OLLAMA_NAMES_PASSES_GEMMA3`

Bench harness controls:

1. `--api-timeout`
2. `--claude-timeout`
3. `--claude-jobs`
4. `--max-pages`
5. `--max-docs`

## 8. What Worked vs What Did Not

Worked reliably:

1. Gemma `format: "json"` + staged extraction.
2. Names double-pass.
3. Targeted regex/context catches for known leak structures.
4. Hard cohort rerun loops with fast patch/verify cycles.

Did not work alone:

1. Prompt-only refinement.
2. Single global temp tuning without deterministic backstops.
3. One-shot broad benchmarks without fail clustering.

## 9. Recommended Operating Mode (Now)

For reliability-first production-like testing:

1. Keep staged extraction enabled.
2. Keep Gemma3 on `format: "json"`.
3. Keep names on multi-pass until no-leak target is met in broad cohort.
4. Keep deterministic post-filters as first-class logic (not optional).
5. Benchmark in two tiers:
   - Tier A: hard-fail cohort (must stay green)
   - Tier B: broader cohort (track fail clusters, patch, rerun)

For speed-first smoke tests:

1. names-only extraction
2. single pass
3. no inline Claude; batch verify later

## 10. Open Issues

1. Throughput remains expensive with 4 extraction calls/doc + verification.
2. OCR-like artifact classes are effectively unbounded; long-tail regex maintenance will continue.
3. Verifier prompt strictness still needs careful balance to avoid false positives.
4. A full 50/50 broad pass was not completed in the final run because the user stopped the run early.

## 11. Practical Next Step If We Resume

Fastest path to 50/50 from current state:

1. Patch the two known fail clusters from the stopped 50-run (`Bpchum` person-field, Lohmaer address variant).
2. Rerun only failed subset first.
3. If clean, rerun full 50.
4. Freeze baseline config + fail cohort as regression gate.

---

## Appendix A: Relevant Files

- `app/endpoints/anonymization.py`
- `anon/anonymization_service.py`
- `tests/pipeline_anonymize_claude_verify.sh`
- `docs/bug-gemma3-structured-output-hang.md`
- `docs/anonymization-tightening.md`
- `docs/plan-anonymization-refactor.md`
