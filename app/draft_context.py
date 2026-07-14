"""Draft-context assembly for /workflow/draft-context and /workflow/verify-facts.

Gives direct (terminal-side) drafting the same context enrichment /generate
injects. The endpoint layer passes in the exact builder callables /generate
uses; this module only handles composition and degradation, so it stays
unit-testable without DB or network."""
from __future__ import annotations

from typing import Callable, Iterable


def _safe_block(fn: Callable[[], str], label: str) -> str:
    """Run one block builder; any failure degrades to '' (same semantics as
    the inline try/excepts in /generate — context must never break drafting)."""
    try:
        return fn() or ""
    except Exception as exc:  # noqa: BLE001 — degradation is the contract
        print(f"[WARN] draft-context block '{label}' failed: {exc}")
        return ""


def assemble_draft_context(
    query: str,
    *,
    rag_block_fn: Callable[[], str],
    statute_block_fn: Callable[[], str],
    memory_block_fn: Callable[[], str],
    style_rules: str,
) -> dict:
    """Compose the four /generate-parity context blocks for `query`."""
    return {
        "rag_block": _safe_block(rag_block_fn, "rag"),
        "statute_block": _safe_block(statute_block_fn, "statutes"),
        "case_memory_block": _safe_block(memory_block_fn, "case_memory"),
        "style_rules": style_rules,
    }


def verify_facts_with_sources(
    text: str, memory_text: str = "", sources: Iterable[str] = ()
) -> dict:
    """verify_facts for terminal drafts: plain source strings instead of the
    selected_documents structure. _fact_corpus concatenates memory + document
    pages, so folding the sources into memory_text is equivalent."""
    from citation_verifier import verify_facts

    corpus = "\n".join([memory_text or "", *[s or "" for s in sources]])
    return verify_facts(text, {}, memory_text=corpus)
