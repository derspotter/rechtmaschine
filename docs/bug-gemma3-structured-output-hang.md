# Bug: Gemma3:12b hangs on structured output with temperature 0.0

## Symptom

File `2024-12-10_5681674_24K2623_7237581-439pdf_Beiakte_004_Anhörung_p114-123.pdf` (10 pages, 8502 chars, native text MARiS-Masken-Informationen dump) consistently times out at 300s during entity extraction via Ollama.

Other files (up to 74K chars) complete in 12-65s. This specific file sometimes completes in 11-18s and sometimes hangs for 300s+, non-deterministically.

## Root cause

Three known Ollama/Gemma3 bugs interact:

1. **`temperature: 0.0` + Gemma3 = pathological slowdown** (Ollama [#11306](https://github.com/ollama/ollama/issues/11306))
   - Introduced in Ollama 0.9.5. Low temperature causes extremely long inference times with Gemma3 specifically.
   - The same request that completes in 22s at default temp times out at 3+ min with low temp.

2. **Structured output memory leak** (Ollama [#10688](https://github.com/ollama/ollama/issues/10688))
   - Gemma3 + JSON schema format leaks ~30MB/s system RAM during inference.
   - Over multiple requests in a bench run, RAM accumulates and can cause hangs.
   - The Ollama process must be killed entirely to reclaim the leaked memory.

3. **GBNF grammar sampling deadlock** (llama.cpp [#2971](https://github.com/ggml-org/llama.cpp/issues/2971))
   - Ollama converts JSON schema to GBNF grammar and masks invalid tokens on CPU (not GPU).
   - With highly repetitive input (this file repeats "Goudarzi, Pourya, 20.12.1992, Andimeschk" dozens of times), the peaked token distribution + grammar constraints can deadlock the sampler.
   - `repeat_penalty` (default 1.1) worsens this by further distorting the distribution.

## Why this file specifically

The document is a MARiS database dump — extremely structured, with the same person data block repeated twice (pages 114-117 and 118-123). The name "Goudarzi, Pourya" appears ~20 times with identical surrounding context. This creates a degenerate sampling distribution where the model wants to keep generating the same name entries, and the grammar constraint + repeat_penalty create a near-deadlock.

## Evidence

- Calling Ollama directly (`curl localhost:11434/api/generate`) with the same payload completes in <60s.
- Calling via service_manager `/extract-entities` from desktop completes in ~26s.
- Calling from inside the Docker container completes in 11s.
- During bench runs (sequential OCR+anon cycles with VRAM switching), the same file hangs for 300s+. The memory leak from prior requests likely contributes.
- The file succeeded in 18s under the OLD architecture (`anonymization_service.py` subprocess with `timeout=120.0`).

## Tested mitigations

| Change | Result |
|--------|--------|
| `temperature: 0.1` (was 0.0) | Fixes the hang. Extraction completes in 60-120s instead of hanging. But much slower than the 11-18s seen with temp 0.0 when it doesn't hang. |
| `num_predict: 2048` | Prevents infinite generation but truncates output mid-JSON for this file (entity JSON is ~4500 chars). |
| `num_predict: 4096` | Output completes but takes 120s+ (model generates many duplicate name entries). |
| `repeat_penalty: 1.0` | Removes one source of sampling conflict. Effect unclear in isolation. |

## Recommended options (current state)

```python
"options": {
    "temperature": 0.1,       # Avoids Gemma3 low-temp bug
    "num_predict": 4096,      # Hard cap, prevents infinite generation
    "num_ctx": 32768,
    "repeat_penalty": 1.0,    # Avoids conflict with grammar on repetitive input
}
```

## Open questions

- Why does `temperature: 0.1` make extraction 5-10x slower (120s vs 11s) even when it doesn't hang? The model may be generating many more tokens with the slightly higher temperature.
- Would a smaller `num_predict` (e.g., 1024) work for most files and only truncate on pathological cases like this one? The entity JSON for non-repetitive files is typically <500 chars.
- Would upgrading Ollama to a newer version fix the memory leak and low-temp bugs?
- Would switching to the OpenAI-compatible endpoint (`/v1/chat/completions`) with `max_tokens` behave differently?

## References

- Ollama [#11306](https://github.com/ollama/ollama/issues/11306) — Low temp + Gemma3 slowdown
- Ollama [#10688](https://github.com/ollama/ollama/issues/10688) — Gemma3 structured output memory leak
- Ollama [#10040](https://github.com/ollama/ollama/issues/10040) — Gemma3 system memory leak (KV cache)
- Ollama [#2805](https://github.com/ollama/ollama/issues/2805) — Infinite generation loop
- llama.cpp [#2971](https://github.com/ggml-org/llama.cpp/issues/2971) — Grammar sampling freeze
