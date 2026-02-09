#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage:
  bash tests/bench_anonymize_pdfs.sh [--dir DIR] [--type TYPE] [--dry-run]

What it does:
  - Finds PDFs in DIRs (default: sample_files)
  - Skips PDFs with >50 pages (page count via pdfinfo)
  - Uploads each remaining PDF to the app endpoint POST /anonymize-file
    The app will run check_pdf_needs_ocr(...) and do OCR if necessary.
  - Writes per-file JSON + anonymized text under tmp/anonymize_bench/
  - Appends a CSV summary to tmp/anonymize_bench/results.csv

Options:
  --dir DIR     Add an input directory (can be repeated). Default: sample_files
  --type TYPE   Force document_type for all files (e.g. "Anhörung" or "Bescheid").
                If omitted, it is guessed from filename.
  --dry-run     Only print what would be processed; do not call the API.

Notes:
  - /anonymize-file is rate-limited (currently 5/hour).
  - Requires: curl, jq, pdfinfo, docker (to mint an auth token from the app container).
EOF
}

dirs=()
forced_type=""
dry_run=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)
      dirs+=("${2:-}")
      shift 2
      ;;
    --type)
      forced_type="${2:-}"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ ${#dirs[@]} -eq 0 ]; then
  dirs=("sample_files")
fi

for bin in curl jq pdfinfo docker; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "Missing required binary: $bin" >&2
    exit 1
  fi
done

mkdir -p tmp/anonymize_bench

TOKEN="$(docker exec rechtmaschine-app bash -lc "python3 - <<'PY'
from auth import create_access_token
print(create_access_token({'sub': 'jay'}))
PY")"

ts="$(date -u +%Y%m%d-%H%M%S)"
csv="tmp/anonymize_bench/results.csv"
if [ ! -f "$csv" ]; then
  echo "timestamp,file,pages,document_type,duration_s,http_status,ocr_used,cached,confidence,input_characters,processed_characters,remaining_characters,placeholders_person,placeholders_ort,placeholders_datum" >> "$csv"
fi

guess_type() {
  local f="$1"
  local lower
  lower="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
  if echo "$lower" | rg -q "anhör|anhoer|niederschrift"; then
    echo "Anhörung"
    return
  fi
  if echo "$lower" | rg -q "bescheid|anlage_k"; then
    echo "Bescheid"
    return
  fi
  echo "Sonstiges"
}

page_count() {
  local f="$1"
  pdfinfo "$f" 2>/dev/null | awk '/^Pages:/ {print $2; exit}'
}

placeholder_count() {
  # Count tokens like [PERSON], [ORT], [DATUM] in anonymized text.
  local token="$1"
  local text="$2"
  # Use grep -o for portable counting.
  printf "%s" "$text" | grep -o "\\[$token\\]" 2>/dev/null | wc -l | tr -d ' '
}

echo "Scanning:"
for d in "${dirs[@]}"; do
  echo "  - $d"
done
echo ""

found_any=0
for d in "${dirs[@]}"; do
  if [ ! -d "$d" ]; then
    echo "Skip missing dir: $d" >&2
    continue
  fi
  while IFS= read -r -d '' f; do
    found_any=1

    pages="$(page_count "$f" || true)"
    if [ -z "${pages:-}" ]; then
      echo "Skip (could not read page count): $f" >&2
      continue
    fi
    if [ "$pages" -gt 50 ]; then
      echo "Skip (>50 pages): $f (pages=$pages)"
      continue
    fi

    doc_type="$forced_type"
    if [ -z "$doc_type" ]; then
      doc_type="$(guess_type "$f")"
    fi

    echo "Process: $f (pages=$pages, type=$doc_type)"
    if [ "$dry_run" = "1" ]; then
      continue
    fi

    base="$(basename "$f")"
    safe_base="$(printf "%s" "$base" | tr ' /' '__' | tr -cd '[:alnum:]_.-')"
    out_json="tmp/anonymize_bench/${ts}__${safe_base}.json"
    out_txt="tmp/anonymize_bench/${ts}__${safe_base}.anonymized.txt"

    # curl: capture status + total time. Use a temp file for the body so we can parse on error too.
    body_tmp="$(mktemp)"
    metrics_tmp="$(mktemp)"
    http_status="$(
      curl -sS -o "$body_tmp" \
        -w "%{http_code} %{time_total}" \
        -X POST "http://127.0.0.1:8000/anonymize-file" \
        -H "Authorization: Bearer ${TOKEN}" \
        -F "document_type=${doc_type}" \
        -F "file=@${f};type=application/pdf"
    )"
    # split: "200 12.345"
    duration_s="$(printf "%s" "$http_status" | awk '{print $2}')"
    http_status="$(printf "%s" "$http_status" | awk '{print $1}')"
    rm -f "$metrics_tmp" || true

    # Best-effort parse for duration and fields; on error, keep raw body.
    if [ "$http_status" != "200" ]; then
      echo "  HTTP $http_status (see $out_json)" >&2
      jq -R -s '{error: true, http_status: env.HTTP_STATUS, body: .}' <"$body_tmp" \
        | HTTP_STATUS="$http_status" jq '.http_status=$ENV.HTTP_STATUS' >"$out_json" || cp "$body_tmp" "$out_json"
      rm -f "$body_tmp"
      continue
    fi

    cp "$body_tmp" "$out_json"
    rm -f "$body_tmp"

    anonymized_text="$(jq -r '.anonymized_text // ""' <"$out_json")"
    printf "%s" "$anonymized_text" >"$out_txt"

    # Extract metrics for CSV.
    ocr_used="$(jq -r '.ocr_used // false' <"$out_json")"
    cached="$(jq -r '.cached // false' <"$out_json")"
    confidence="$(jq -r '.confidence // 0' <"$out_json")"
    input_characters="$(jq -r '.input_characters // 0' <"$out_json")"
    processed_characters="$(jq -r '.processed_characters // 0' <"$out_json")"
    remaining_characters="$(jq -r '.remaining_characters // 0' <"$out_json")"

    persons="$(placeholder_count PERSON "$anonymized_text")"
    ort="$(placeholder_count ORT "$anonymized_text")"
    datum="$(placeholder_count DATUM "$anonymized_text")"

    echo "${ts},${base},${pages},${doc_type},${duration_s},${http_status},${ocr_used},${cached},${confidence},${input_characters},${processed_characters},${remaining_characters},${persons},${ort},${datum}" >> "$csv"
    echo "  -> wrote $out_json"
    echo "  -> wrote $out_txt"
  done < <(find "$d" -maxdepth 2 -type f -name '*.pdf' -print0 | sort -z)
done

if [ "$found_any" = "0" ]; then
  echo "No PDFs found in specified dirs." >&2
  exit 1
fi

echo ""
echo "Done. Summary CSV: $csv"
echo "Outputs: tmp/anonymize_bench/"
