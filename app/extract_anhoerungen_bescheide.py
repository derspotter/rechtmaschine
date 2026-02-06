#!/usr/bin/env python3
import re
import unicodedata
from typing import List

DEFAULT_INCLUDES = [
    "Bescheid",
    "Bescheid_Ablehnung",
    "Anhörung_Standard_Mann",
    "Anhörung_Standard_Frau",
    "Anhörung_Zulässigkeit_Mann",
    "Anhörung_Zulässigkeit_Frau",
    "Erstbefragung_Zulässigkeit",
]


def normalize_text(text: str) -> str:
    # Strip diacritics so "Anhörung" becomes "Anhorung"
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    ).lower()


def iter_outline_items(items):
    for item in items:
        yield item
        if getattr(item, "children", None):
            yield from iter_outline_items(item.children)


def get_page_index(pdf, item):
    dest = item.destination
    if dest is None and item.action is not None:
        try:
            dest = item.action.get("/D")
        except Exception:
            dest = None
    if dest is None:
        return None
    try:
        page_ref = dest[0]
    except Exception:
        return None
    try:
        return pdf.pages.index(page_ref)
    except Exception:
        if isinstance(page_ref, int):
            if 0 <= page_ref < len(pdf.pages):
                return page_ref
            if 1 <= page_ref <= len(pdf.pages):
                return page_ref - 1
    return None


def collect_outline_items(pdf):
    with pdf.open_outline() as outline:
        items = []
        order = 0
        for item in iter_outline_items(outline.root):
            title = item.title or ""
            start = get_page_index(pdf, item)
            if start is None:
                continue
            items.append({"order": order, "title": title, "start": start})
            order += 1
        items.sort(key=lambda x: (x["start"], x["order"]))
        for i, it in enumerate(items):
            if i + 1 < len(items):
                end = items[i + 1]["start"] - 1
            else:
                end = len(pdf.pages) - 1
            if end < it["start"]:
                end = it["start"]
            it["end"] = end
        return items


def select_items(items, pattern: str):
    rx = re.compile(pattern)
    selected = []
    for it in items:
        if rx.search(normalize_text(it["title"])):
            selected.append(it)
    return selected


def select_items_by_includes(items, includes: List[str]):
    wanted = [normalize_text(x) for x in includes]
    selected = []
    for it in items:
        title_n = normalize_text(it["title"])
        matched = False
        for w in wanted:
            if not w:
                continue
            if w == "bescheid":
                if re.search(r"\bbescheid\b$", title_n):
                    matched = True
                    break
                continue
            if w in title_n:
                matched = True
                break
        if matched:
            selected.append(it)
    return selected
