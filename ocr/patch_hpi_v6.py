#!/usr/bin/env python3
"""Idempotently register PP-OCRv6 models in PaddleX's HPI backend registry.

PaddleOCR 3.7 ships PP-OCRv6 as the default models, but the bundled HPI
plugin (ultra-infer / paddlex hpi_model_info_collection.json) doesn't list
them yet, so enable_hpi=True fails with "<model> is not a known model".
This copies each PP-OCRv5 model's supported-backend list onto its v6
equivalent. Re-run after any paddlex reinstall/upgrade. Safe to run twice.

Usage:  <venv>/bin/python patch_hpi_v6.py
"""
import json, copy, sys
from importlib.util import find_spec

spec = find_spec("paddlex")
if not spec:
    sys.exit("paddlex not importable in this interpreter")
root = spec.submodule_search_locations[0]
p = f"{root}/inference/models/runners/hpi/hpi_model_info_collection.json"

PAIRS = [
    ("PP-OCRv5_server_det", "PP-OCRv6_medium_det"),
    ("PP-OCRv5_server_det", "PP-OCRv6_small_det"),
    ("PP-OCRv5_server_det", "PP-OCRv6_tiny_det"),
    ("PP-OCRv5_server_rec", "PP-OCRv6_medium_rec"),
    ("PP-OCRv5_server_rec", "PP-OCRv6_small_rec"),
    ("PP-OCRv5_server_rec", "PP-OCRv6_tiny_rec"),
]

d = json.load(open(p))
added = 0
for _plat, vers in d.items():
    if not isinstance(vers, dict):
        continue
    for _ver, models in vers.items():
        if not isinstance(models, dict):
            continue
        for v5, v6 in PAIRS:
            if v5 in models and v6 not in models:
                models[v6] = copy.deepcopy(models[v5])
                added += 1
json.dump(d, open(p, "w"), indent=1)
print(f"patched {p}\nadded {added} PP-OCRv6 HPI entries")
