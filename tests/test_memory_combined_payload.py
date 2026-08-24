"""Regression test for the slimmed /memory/cases/{id} payload (`_combined_payload`).

Background: the endpoint used to emit the case brief twice ("brief" and
"case_brief", byte-identical, each built by its own render call) and to carry
a `search_text` that mirrors `rendered` verbatim — ~383 KB for a grown case,
~66 KB of it substance (measured on 096/26, 2026-08-24). The fix renders each
target once, drops the legacy "brief" alias (the frontend reads
`data.case_brief || data.brief`, app/static/js/app.js), and ships
`search_text` only when it differs from `rendered`.

The endpoint module needs the full FastAPI stack, so this checks the shape of
`_combined_payload` at source level — enough to catch the alias or the double
render coming back. Run from the repo root:

    python3 tests/test_memory_combined_payload.py
"""

import sys
from pathlib import Path


def main():
    src = (Path(__file__).resolve().parents[1] / "app" / "endpoints" / "agent_memory.py").read_text()
    payload_src = src.split("def _combined_payload", 1)[1].split("\ndef ", 1)[0]

    checks = [
        ('"brief":' not in payload_src,
         'legacy "brief" alias is back in _combined_payload'),
        (payload_src.count("render_case_brief_compact") == 1,
         "case brief must be rendered exactly once"),
        (payload_src.count("render_case_strategy_compact") == 1,
         "case strategy must be rendered exactly once"),
        ('"case_brief"' in payload_src and '"case_strategy"' in payload_src,
         "case_brief/case_strategy keys missing"),
        ('payload.pop("search_text"' in payload_src,
         "search_text dedup guard missing"),
        ('"memory"' in payload_src,
         'nested "memory" key removed — frontend app.js reads data.memory first'),
    ]

    failed = [msg for ok, msg in checks if not ok]
    for msg in failed:
        print("FAIL", msg)
    if not failed:
        print("ok   single render per target, no brief alias, search_text guarded, memory key kept")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
