# Codex-Review (gpt-6-astra, reasoning high) der Branch 570e5c6..7515c1c, 07.09.2026

Brief: siehe Session-Scratchpad; Antwort unverändert, Duplikat des Schlussblocks entfernt.

**I would not merge this yet.** The happy path works, but I reproduced failures in rebase, citation blocking, wiki filtering, and cloud-context provenance.

Reviewed `570e5c6 → 7515c1c` and the supplied skills diff. No files changed. Locally, **59 focused app tests and five hook tests passed**, as did the triage assertion script. Running the registry tests before the wiki tests produced **nine failures from leaked module stubs**. The route-test run stalled and was interrupted. I did not rerun live acceptance or database migrations.

**Critical**

No confirmed Critical finding. The Important findings below are sufficient to block merging.

**Important**

1. **Rebase can turn a replacement into a deletion.**  
   [app/assessment_memory.py:518](/var/opt/docker/rechtmaschine/app/assessment_memory.py:518)

   A valid proposal contains `remove /gutachten/by-id/aa`, followed by `append /gutachten/-` with replacement `aa`. After an unrelated `bb` change, rebase keeps the removal but drops the append because `existing_ids` still contains `aa`. Accepting the rebased proposal deletes the Gutachten.

   **Reproduced:** original ops produce the replacement, rebased ops contain only `remove`, resulting content has no Gutachten.

   **Fix:** filter genuine conflicts, then apply all surviving operations together against the new content. Supersede the proposal if that complete application fails. Do not validate and discard operations incrementally against a static identity set.

   **This is worse than your known rebase limitation:** HEAD does not consistently supersede the whole failed proposal. It can preserve a destructive fragment.

2. **The cloud-facing context export contains unpseudonymized assessment metadata.**  
   [app/agent_memory_service.py:1262](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:1262), [app/endpoints/workflow.py:275](/var/opt/docker/rechtmaschine/app/endpoints/workflow.py:275)

   `assessment_blocked_az` comes from raw assessment fields, after only a truthiness check on the pseudonymized text. `az` accepts arbitrary strings up to 80 characters.

   **Reproduced:** with `az="9 K 1/26 (Max Mustermann)"`, the pseudonymized prompt contains no name, but `grounding.assessment_blocked_az` still contains `Max Mustermann`. `/draft-context` returns that grounding dictionary to the terminal drafting session.

   I did **not** find the internal generation path sending the entire grounding dictionary to its cloud model. The concrete exposure is the context export consumed by cloud-backed sessions.

   **Fix:** keep raw citation metadata server-local. Export validated canonical Az without annotations, and pseudonymize any remaining textual provenance. Treat the complete draft-context response as the cloud boundary.

3. **The wiki identifying-data gate does not cover its new primary source.**  
   [app/endpoints/pattern_wiki.py:167](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:167), [app/endpoints/pattern_wiki.py:215](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:215)

   `_forbidden_tokens` reads brief and strategy, never assessment content. A birth date, identifier, or third-party name appearing only in a Gutachten can survive extraction and be persisted in a firm-scoped wiki entry.

   There are additional inherited weaknesses now exposed to assessment input:

   - Matching is case-sensitive. `mustermann` survives a forbidden token `Mustermann`.
   - The citation exemption covers arbitrary text between the court and decision type. I reproduced `VG Teststadt, Frau Mustermann, Urteil vom … – 18 E 491/12` escaping the name gate.

   Pending status prevents automatic prompt injection until activation, but does not prevent persistence or firm visibility.

   **Fix:** derive identifying tokens from every input source and known case entities. Compare normalized text. Exempt only recognized citation components, never an arbitrary citation-shaped span containing names.

4. **The wiki whitelist is neither comprehensive nor applied to every persisted field.**  
   [app/endpoints/pattern_wiki.py:232](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:232), [app/endpoints/pattern_wiki.py:342](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:342)

   **Reproduced with an empty whitelist:** `EuGH C-151/22`, `EGMR Nr. 12345/19`, and compact `18E491/12` survive unchanged.

   Independently, stripping skips `title`, `tags`, and `fingerprint`. A foreign `5 K 9/23` placed there survives unless it happens to match an identifying token from the source case.

   The latest fixes also leave false positives:

   - `Art. 3 RL 2011/95` becomes `Art.`
   - A correctly whitelisted `EuGH, Urteil vom 21.09.2023 – C-151/22` is rejected by the identifying-data gate when that decision date is among the forbidden tokens. Its exemption regex does not understand this EuGH Az.
   - The bare fallback accepts a prefixed Az when only its unprefixed core is whitelisted. `M 10 K 21.3767` survives a whitelist containing only `10 K 21.3767`, despite different canonical keys.

   **Fix:** use one citation parser and canonical identity throughout stripping and exemptions, traverse every persisted string field, and recognize norm spans before attempting Az extraction. Remove the permissive prefix/core equivalence.

5. **“Nicht zitierfähig” is bypassed by both parser gaps and the terminal verification route.**  
   [app/citation_verifier.py:136](/var/opt/docker/rechtmaschine/app/citation_verifier.py:136), [app/endpoints/workflow.py:302](/var/opt/docker/rechtmaschine/app/endpoints/workflow.py:302)

   **Reproduced:** blocked `10 ZB 22.1187`, `M 10 K 21.3767`, `6 Bs 176/25`, and `C-151/22` produce no fact warning. Prefixes and trailing court designators can also prevent blocklist equality.

   More seriously, `/workflow/verify-facts` neither collects nor forwards blocked Az. Even ordinary `18 E 491/12` passes when its sole support is the “Nicht zitierfähig” line itself.

   **Fix:** propagate the blocklist through `verify_facts_with_sources`, use canonical citation identities, and remove blocklist lines from the supporting corpus as specified. Check blocked citations even when the remaining corpus is empty.

   A model can still generate a prohibited citation regardless of prompt instructions. The promised deterministic warning must work on every drafting path.

6. **Stale proposals can acquire a fresh version without resolving their original conflict.**  
   [app/agent_memory_service.py:660](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:660), [app/agent_memory_service.py:865](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:865)

   Creation dry-runs against current content but never checks `expected_version`.

   Scenario: A changed between v1 and v2. A session then creates a stale v1 replacement for A. An earlier pending proposal changes B and advances the target to v3. Rebase considers only the B change, preserves the stale A replacement, and changes its expected version to v3. It can now overwrite the v2 correction.

   **Fix:** validate the creation version while holding the target lock through proposal insertion, or retain the actual base snapshot and perform a three-way rebase. Merely updating `expected_version` is insufficient.

   The existing same-target ordering check and explicit `force` override are otherwise present. Version mismatches currently return 400, while ordering violations return 409.

7. **An unchanged `set by-id` silently destroys server-owned verification state.**  
   [app/assessment_memory.py:235](/var/opt/docker/rechtmaschine/app/assessment_memory.py:235), [app/agent_memory_service.py:971](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:971)

   Applying `set` strips store fields and materializes `unchecked`. Comparing substantive content then reports no changed IDs, so reconciliation skips that Gutachten.

   **Reproduced:** a verified Fundstelle becomes `unchecked`, loses its store ID and timestamp, and returns **no warnings**.

   **Fix:** reconcile every Gutachten whose fields were reset by append/set, even when its substantive content is unchanged. Alternatively, preserve server fields for a true no-op.

   I found no direct route to forge `verified` through ordinary client ops. Clobbering legitimate server state is the demonstrated defect.

8. **Size-validation failures are swallowed as store outages.**  
   [app/agent_memory_service.py:979](/var/opt/docker/rechtmaschine/app/agent_memory_service.py:979)

   `reconcile_store` validates the enriched content inside the broad exception handler. A valid near-limit Gutachten can exceed 24 kB after adding server timestamps and IDs.

   **Reproduced:** a 23,989-byte validated Gutachten fails enrichment. Accept catches that validation error and proceeds with its unenriched `unchecked` content.

   **Fix:** catch technical store-read failures separately. Let enriched-content validation produce the specified 400, or reserve sufficient server-field space during proposal creation.

   I did **not** find a final-content size-limit bypass through rebase. The problem is silent degradation and acceptance of something whose required reconciliation failed validation.

9. **The nightly recheck can exhaust its quota on cases without Gutachten.**  
   [memory_triage.py:598](/home/jay/kanzlei/skills/rechtmaschine/scripts/memory_triage.py:598), [app/endpoints/agent_memory.py:1973](/var/opt/docker/rechtmaschine/app/endpoints/agent_memory.py:1973)

   Every successfully triaged case is rechecked, including cases with no pending proposals and no assessment. The endpoint allows 60 calls/hour per token. A sufficiently large nightly run consumes the quota before reaching later assessment cases.

   Failures are logged and the job still returns success. Store deactivations can therefore remain unapplied while the nightly looks successful.

   **Fix:** select only cases with active Gutachten. Prefer a batched owner-scoped recheck that loads the store once, and report incomplete rechecks explicitly.

10. **The Stop-hook deadline does not bound execution, and ownership lookup failures are treated as “own.”**  
    [memory_hygiene_hook.py:126](/home/jay/kanzlei/skills/rechtmaschine/scripts/memory_hygiene_hook.py:126), [memory_triage.py:120](/home/jay/kanzlei/skills/rechtmaschine/scripts/memory_triage.py:120), [memory_triage.py:403](/home/jay/kanzlei/skills/rechtmaschine/scripts/memory_triage.py:403)

    The 12-second deadline is checked before starting a case. Ownership lookup, case resolution, and proposal retrieval each retain their independent 60-second timeout. A successful but slow sequence can substantially exceed the shared budget.

    Also, `case_owner` converts lookup errors to `None`, and `is_foreign_case` interprets that as eligible. The new hook’s exception guard never sees that failure and may proceed to read a foreign case.

    **Fix:** pass the remaining deadline into every subprocess and recheck it between calls. Distinguish confirmed ownership, confirmed foreign ownership, and lookup failure. Skip unresolved ownership.

**Minor**

- **Historical proposals permanently suppress new Vermerk reminders.**  
  [memory_hygiene_hook.py:145](/home/jay/kanzlei/skills/rechtmaschine/scripts/memory_hygiene_hook.py:145) matches any returned session assessment proposal, including old rejected ones. **Fix:** require creation after the session baseline, as specified. The underlying 50-row proposal cap also means “all statuses” is not a complete history.

- **`/notizen` has no conflict handling.**  
  [app/assessment_memory.py:524](/var/opt/docker/rechtmaschine/app/assessment_memory.py:524) retains scalar sets unconditionally, so stale notes can overwrite newly accepted notes. **Fix:** include scalar changes in the conflict scope and validate the complete surviving patch.

- **Wiki truncation can discard every assessment while claiming assessment provenance.**  
  [app/assessment_memory.py:665](/var/opt/docker/rechtmaschine/app/assessment_memory.py:665) breaks when the first rendered Gutachten exceeds the budget. I reproduced output consisting solely of `[weitere Gutachten gekürzt: 1]`. The whitelist still includes its citations. **Fix:** use staged reduction and return the IDs and whitelist of actually rendered assessments. Deduplicate or bound references within each Prüfungspunkt.

- **Cloud truncation removes detail from all Gutachten together.**  
  [app/assessment_memory.py:356](/var/opt/docker/rechtmaschine/app/assessment_memory.py:356) drops all risks, then all Prüfung detail, instead of reducing the oldest entry first. A custom small budget is also exceeded by the final retained entry. **Fix:** implement the specified per-entry reduction and a hard final budget.

- **Successful unchanged rechecks do not update `store_checked_at`.**  
  [app/assessment_memory.py:591](/var/opt/docker/rechtmaschine/app/assessment_memory.py:591) writes only when state or entry ID changes. **Fix:** persist the check timestamp separately from the change counters and follow the specified revision policy.

- **Duplicate Fundstellen are ambiguous.**  
  [app/assessment_memory.py:294](/var/opt/docker/rechtmaschine/app/assessment_memory.py:294), [app/assessment_memory.py:553](/var/opt/docker/rechtmaschine/app/assessment_memory.py:553) collapse citations by raw Az without validating uniqueness. Two dates/states for the same Az can disagree between rendering, blocking, and change counting. **Fix:** enforce an unambiguous citation identity within each Gutachten.

- **Store scans load unnecessary data and choose duplicate matches nondeterministically.**  
  [app/assessment_memory.py:390](/var/opt/docker/rechtmaschine/app/assessment_memory.py:390) loads complete ORM entries, including large text/JSON fields, for every recheck. **Fix:** select only ID, Az, and date, and order equivalent matches deterministically.

- **Accept warnings disappear immediately in the UI.**  
  [app/static/js/app.js:1789](/var/opt/docker/rechtmaschine/app/static/js/app.js:1789) sets warning text, then calls `loadCaseMemory()`, which replaces it with loading/success text. **Fix:** retain warnings in a dedicated element or refresh silently before displaying them.

- **Wiki source provenance omits Gutachten IDs.**  
  [app/endpoints/pattern_wiki.py:401](/var/opt/docker/rechtmaschine/app/endpoints/pattern_wiki.py:401) records the case and generic source type only. **Fix:** persist the actual contributing assessment IDs and versions.

- **The migration creates a redundant source index.**  
  [app/main.py:1061](/var/opt/docker/rechtmaschine/app/main.py:1061) uses a different name from the ORM-generated index on the same FK. ORM and SQL defaults also differ for version/timestamps. **Fix:** align index names and intended defaults.

- **Registry tests contaminate subsequent tests.**  
  [tests/test_memory_target_registry.py:18](/var/opt/docker/rechtmaschine/tests/test_memory_target_registry.py:18) installs global stubs without restoring them. Running this file before the wiki tests produced **9 failed, 6 passed**, including missing `sqlalchemy.create_engine`. **Fix:** isolate stubs and imported modules, or use the real dependencies.

Store reconciliation otherwise implements the chosen weak semantics: any active normalized-Az/date match verifies, missing dates produce a mismatch, and court identity is ignored. Duplicate Az across courts are consequently a limitation of that accepted contract, not a newly discovered stronger verification guarantee.

The migration contains no destructive row rewrite. Its recorded-name skip and `IF NOT EXISTS` statements support normal reruns, but they do not repair an existing incompatible table. The supplied migration test does not establish upgrade correctness.

## What you are missing

**The tests mostly cover helpers, not the boundaries where the failures occur.**

| Test area | What it does not establish |
|---|---|
| [Store tests:55](/var/opt/docker/rechtmaschine/tests/test_memory_assessment_store.py:55) | Actual accept transaction, isolated store failure, enriched-size rejection, post-commit spawn ordering, revision consistency |
| [Rebase test:57](/var/opt/docker/rechtmaschine/tests/test_memory_assessment_rebase.py:57) | Semantic preservation of a multi-op patch. It checks operation count for append/set, not remove/replacement or size-dependent sequences |
| [Prompt tests:183](/var/opt/docker/rechtmaschine/tests/test_memory_assessment_prompt_context.py:183) | Real entity replacement or sanitization of the complete exported provenance |
| [Wiki tests:70](/var/opt/docker/rechtmaschine/tests/test_pattern_wiki_distill_assessment.py:70) | Full extraction-to-persistence flow, all string fields, assessment-only PII, or combined whitelist/PII behavior |
| [Migration test:51](/var/opt/docker/rechtmaschine/tests/test_memory_assessment_migration.py:51) | Fresh installation, upgrade with existing rows, rerun, defaults, cascade behavior. It only inspects existing tables and skips connection failures |
| [Hook budget test:251](/home/jay/kanzlei/skills/rechtmaschine/tests/test_memory_hygiene_hook.py:251) | Actual elapsed-time enforcement. Both workers are mocked and it only compares their deadline arguments |

The simpler design is to centralize **citation extraction, canonical identity, and recognized spans**, then reuse that result for blocking, whitelist stripping, and identifying-data exemptions. The current collection of incompatible regexes is the main source of safety gaps.

Also missing is downstream invalidation: rechecking an assessment does not revoke citations already distilled into an active wiki entry. Store verification is a snapshot, and the wiki currently loses enough provenance to make later correction difficult.

Your live acceptance is useful evidence for the normal path. It does not cover the demonstrated deletion, metadata leak, or citation-filter failures.

**Merge verdict: do not ship.**
