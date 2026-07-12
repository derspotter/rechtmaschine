#!/usr/bin/env bash
# Fleet-wide case-memory report (read-only):
#  1. consolidation jobs of the last 24h (completed / skipped / failed, per case)
#  2. list sizes per case (consolidation-threshold view)
#  3. field-overlap analysis between the four redundant brief/strategy field pairs
# Usage: ./memory-report.sh [--since-hours N]
set -euo pipefail
HOURS="${2:-24}"; [ "${1:-}" = "--since-hours" ] || HOURS=24

PSQL=(docker exec rechtmaschine-postgres psql -U rechtmaschine -d rechtmaschine_db -tA -F' | ')

echo "===== 1. Konsolidierungs-Jobs (letzte ${HOURS}h) ====="
"${PSQL[@]}" -c "
select to_char(j.available_at,'MM-DD HH24:MI'), left(c.name,34), j.status,
       coalesce(j.result_payload->>'created','-'),
       left(coalesce(j.result_payload->>'skipped', j.error_message, ''),60)
from memory_reflection_jobs j join cases c on c.id=j.case_id
where j.request_payload->>'trigger'='consolidate'
  and j.available_at > now() - interval '${HOURS} hours'
order by j.available_at;"

echo; echo "===== 2. Listengrößen je Fall (Schwelle 10) ====="
"${PSQL[@]}" -c "
select left(c.name,34),
       coalesce(jsonb_array_length(cb.content_json->'verfahrensstand'),0) vs,
       coalesce(jsonb_array_length(cb.content_json->'sachverhalt'),0) sv,
       coalesce(jsonb_array_length(cb.content_json->'beweismittel'),0) bw,
       coalesce(jsonb_array_length(cb.content_json->'offene_fragen'),0) of,
       coalesce(jsonb_array_length(cs.content_json->'prozessuale_schritte'),0) ps,
       length(cb.content_json::text) chars
from cases c
join case_briefs cb on cb.case_id=c.id
left join case_strategies cs on cs.case_id=c.id
where not coalesce(c.archived,false) and length(cb.content_json::text)>200
order by chars desc;"

echo; echo "===== 3. Feld-Überlappung Brief vs. Strategie (Ähnlichkeit > 0.7) ====="
docker exec -i rechtmaschine-job-worker python3 - <<'PY'
from database import SessionLocal
from models import Case
from difflib import SequenceMatcher
import json
from sqlalchemy import text as _t

PAIRS = [
    ("offene_fragen", "brief", "offene_fragen", "strategy"),
    ("risiken", "brief", "risiken_und_gegenargumente", "strategy"),
    ("verfahrensstand", "brief", "prozessuale_schritte", "strategy"),
    ("beweismittel", "brief", "beweisstrategie", "strategy"),
]

def entries(content, field):
    out = []
    for v in (content or {}).get(field) or []:
        out.append((v.get("name") if isinstance(v, dict) else str(v)) or "")
    return [e for e in out if e]

def overlap(a, b):
    hits = 0
    for x in a:
        for y in b:
            if SequenceMatcher(None, x.lower(), y.lower()).quick_ratio() > 0.7 and \
               SequenceMatcher(None, x.lower(), y.lower()).ratio() > 0.7:
                hits += 1
                break
    return hits

s = SessionLocal()
try:
    rows = s.execute(_t("""
        select c.name, cb.content_json brief, cs.content_json strat
        from cases c
        join case_briefs cb on cb.case_id=c.id
        left join case_strategies cs on cs.case_id=c.id
        where not coalesce(c.archived,false) and length(cb.content_json::text)>200
    """)).fetchall()
    totals = {p[0]+"~"+p[2]: [0, 0] for p in PAIRS}
    print(f"{'Fall':36} " + "  ".join(f"{p[0][:12]}~{p[2][:14]}" for p in PAIRS))
    for name, brief, strat in sorted(rows):
        cells = []
        for bf, _, sf, _ in PAIRS:
            a = entries(brief, bf)
            b = entries(strat, sf)
            h = overlap(a, b) if a and b else 0
            key = bf+"~"+sf
            totals[key][0] += h
            totals[key][1] += min(len(a), len(b)) if a and b else 0
            cells.append(f"{h}/{min(len(a),len(b)) if a and b else 0}")
        if any(c.split('/')[1] != '0' for c in cells):
            print(f"{name[:36]:36} " + "  ".join(f"{c:>27}" for c in cells))
    print("-" * 100)
    print("SUMME überlappend/möglich: " + "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in totals.items()))
finally:
    s.close()
PY
