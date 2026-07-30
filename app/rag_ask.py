"""Kernlogik für den Wissensbasis-Chat: Quellen nummerieren, belegpflichtigen
Prompt bauen, [Qn]-Post-Check. Pure Funktionen — Retrieval und LLM-Aufrufe
bleiben im Endpoint (endpoints/rag.py)."""
import re
from typing import Any, Dict, List, Optional, Tuple

RAG_COLLECTIONS: Tuple[str, ...] = ("kanzlei", "jurisprudence", "doktrin")

_CITATION_RE = re.compile(r"\[Q\d+\]")

SYSTEM_INSTRUCTION = (
    "Du bist der Wissensbasis-Assistent einer Anwaltskanzlei für Migrations- und Sozialrecht. "
    "Beantworte die Frage AUSSCHLIESSLICH auf Grundlage der nummerierten Quellen [Q1]..[Qn]. "
    "Belege jede tatsächliche und rechtliche Aussage mit dem Marker der tragenden Quelle, z. B. [Q2]. "
    "Wörtliche Übernahmen stehen in Anführungszeichen mit Marker. "
    "Was die Quellen nicht tragen, kennzeichne ausdrücklich als 'im Bestand nicht belegt' "
    "und ergänze es nicht aus eigenem Wissen. "
    "Verwende keine Semikola. Antworte auf Deutsch."
)


def number_chunks(chunks_by_collection: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    numbered: List[Dict[str, Any]] = []
    index = 1
    for collection, chunks in chunks_by_collection.items():
        for chunk in chunks or []:
            numbered.append({
                "label": f"Q{index}",
                "collection": collection,
                "header": str(chunk.get("context_header") or ""),
                "text": str(chunk.get("text") or ""),
                "score": float(chunk.get("score") or 0.0),
                "chunk_id": str(chunk.get("chunk_id") or ""),
            })
            index += 1
    return numbered


def trim_history(history: Optional[List[Dict[str, Any]]], max_messages: int = 12) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for msg in history or []:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            cleaned.append({"role": role, "content": content[:4000]})
    return cleaned[-max_messages:]


def build_ask_prompt(
    question: str,
    sources: List[Dict[str, Any]],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    source_blocks = []
    for src in sources:
        origin = f"{src['collection']} {src['header']}".strip()
        source_blocks.append(f"[{src['label']}] ({origin})\n{src['text']}")

    history_block = ""
    trimmed = trim_history(history)
    if trimmed:
        lines = ["BISHERIGER VERLAUF (für Folgefragen):"]
        for msg in trimmed:
            speaker = "Nutzer" if msg["role"] == "user" else "Assistent"
            lines.append(f"{speaker}: {msg['content']}")
        history_block = "\n".join(lines) + "\n\n"

    prompt = (
        "QUELLEN:\n\n" + "\n\n".join(source_blocks) + "\n\n"
        + history_block
        + f"AKTUELLE FRAGE: {question}"
    )
    return SYSTEM_INSTRUCTION, prompt


def has_citation_markers(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def build_chat_messages(
    question: str,
    answer: str,
    model: str,
    sources: List[Dict[str, Any]],
    now_iso: str,
) -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": question, "created_at": now_iso},
        {
            "role": "assistant",
            "content": answer,
            "model": model,
            "sources": sources,
            "unbelegt": not has_citation_markers(answer),
            "created_at": now_iso,
        },
    ]
