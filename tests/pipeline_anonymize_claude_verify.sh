#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
    cat <<'HELP'
Usage:
  bash tests/pipeline_anonymize_claude_verify.sh [options]

What it does:
  - Finds PDFs in one or more directories
  - Enforces page limit before anonymization (default: <=50 pages)
  - Sends each allowed PDF to POST /anonymize-file
  - Verifies each anonymized output with Claude CLI (keyless, Max session)
  - Writes per-file artifacts + run summaries under tmp/anonymize_pipeline/<run_id>/

Options:
  --dir DIR           Input directory (repeatable). Default: extracted_documents/gpt-5-mini
  --max-docs N        Maximum number of PDFs to process. Default: 50
  --max-pages N       Hard page cap. Files above this are skipped. Default: 50
  --claude-timeout N  Timeout in seconds for each Claude verification. Default: 45
  --type TYPE         Force document_type for all files
  --run-id ID         Custom run id (default UTC timestamp)
  --resume            Resume an existing run-id directory (skip already processed files)
  --dry-run           Select + page-filter only, do not call APIs
  -h, --help          Show this help

Required tools:
  curl, jq, pdfinfo, docker, claude, timeout
HELP
}

dirs=()
max_docs=50
max_pages=50
claude_timeout_s=45
forced_type=""
run_id=""
resume=0
dry_run=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir)
            dirs+=("${2:-}")
            shift 2
            ;;
        --max-docs)
            max_docs="${2:-}"
            shift 2
            ;;
        --max-pages)
            max_pages="${2:-}"
            shift 2
            ;;
        --claude-timeout)
            claude_timeout_s="${2:-}"
            shift 2
            ;;
        --type)
            forced_type="${2:-}"
            shift 2
            ;;
        --run-id)
            run_id="${2:-}"
            shift 2
            ;;
        --resume)
            resume=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
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
    dirs=("extracted_documents/gpt-5-mini")
fi

for bin in curl jq pdfinfo docker claude timeout; do
    if ! command -v "$bin" >/dev/null 2>&1; then
        echo "Missing required binary: $bin" >&2
        exit 1
    fi
done

if ! [[ "$max_docs" =~ ^[0-9]+$ ]] || [ "$max_docs" -le 0 ]; then
    echo "--max-docs must be a positive integer" >&2
    exit 2
fi
if ! [[ "$max_pages" =~ ^[0-9]+$ ]] || [ "$max_pages" -le 0 ]; then
    echo "--max-pages must be a positive integer" >&2
    exit 2
fi
if ! [[ "$claude_timeout_s" =~ ^[0-9]+$ ]] || [ "$claude_timeout_s" -le 0 ]; then
    echo "--claude-timeout must be a positive integer" >&2
    exit 2
fi

if [ -z "$run_id" ]; then
    run_id="$(date -u +%Y%m%d-%H%M%S)"
fi

run_dir="tmp/anonymize_pipeline/${run_id}"
processed_file="${run_dir}/processed_paths.txt"
summary_csv="${run_dir}/summary.csv"
summary_jsonl="${run_dir}/summary.jsonl"
skipped_csv="${run_dir}/skipped.csv"
meta_json="${run_dir}/run_meta.json"

if [ -d "$run_dir" ] && [ "$resume" = "0" ]; then
    echo "Run directory already exists: $run_dir" >&2
    echo "Use --resume with the same --run-id to continue." >&2
    exit 2
fi

mkdir -p "$run_dir"
touch "$processed_file"

if [ ! -f "$summary_csv" ]; then
    echo "index,file,pages,document_type,http_status,duration_s,ocr_used,confidence,input_characters,processed_characters,remaining_characters,claude_result,claude_leaks,claude_note,error" > "$summary_csv"
fi
if [ ! -f "$skipped_csv" ]; then
    echo "file,pages,reason" > "$skipped_csv"
fi

trim_ws() {
    echo "$1" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'
}

csv_escape() {
    local s="$1"
    s="${s//\"/\"\"}"
    printf '"%s"' "$s"
}

safe_name() {
    printf "%s" "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

guess_type() {
    local f="$1"
    local lower
    lower="$(basename "$f" | tr '[:upper:]' '[:lower:]')"
    if echo "$lower" | grep -E -q "anhör|anhoer|niederschrift"; then
        echo "Anhörung"
        return
    fi
    if echo "$lower" | grep -E -q "bescheid|anlage_k"; then
        echo "Bescheid"
        return
    fi
    echo "Sonstiges"
}

page_count() {
    local f="$1"
    pdfinfo "$f" 2>/dev/null | awk '/^Pages:/ {print $2; exit}'
}

mint_token() {
    docker exec rechtmaschine-app bash -lc "python3 - <<'PY'
from datetime import timedelta
from auth import create_access_token
print(create_access_token({'sub': 'jay'}, expires_delta=timedelta(hours=8)))
PY"
}

jsonl_write() {
    local idx="$1"
    local file="$2"
    local pages="$3"
    local doc_type="$4"
    local http_status="$5"
    local duration_s="$6"
    local ocr_used="$7"
    local confidence="$8"
    local input_characters="$9"
    local processed_characters="${10}"
    local remaining_characters="${11}"
    local claude_result="${12}"
    local claude_leaks="${13}"
    local claude_note="${14}"
    local error_msg="${15}"

    jq -cn \
        --arg index "$idx" \
        --arg file "$file" \
        --arg pages "$pages" \
        --arg document_type "$doc_type" \
        --arg http_status "$http_status" \
        --arg duration_s "$duration_s" \
        --arg ocr_used "$ocr_used" \
        --arg confidence "$confidence" \
        --arg input_characters "$input_characters" \
        --arg processed_characters "$processed_characters" \
        --arg remaining_characters "$remaining_characters" \
        --arg claude_result "$claude_result" \
        --arg claude_leaks "$claude_leaks" \
        --arg claude_note "$claude_note" \
        --arg error "$error_msg" \
        '{
            index: ($index|tonumber),
            file: $file,
            pages: ($pages|tonumber),
            document_type: $document_type,
            http_status: ($http_status|tonumber),
            duration_s: ($duration_s|tonumber),
            ocr_used: ($ocr_used == "true"),
            confidence: ($confidence|tonumber),
            input_characters: ($input_characters|tonumber),
            processed_characters: ($processed_characters|tonumber),
            remaining_characters: ($remaining_characters|tonumber),
            claude_result: $claude_result,
            claude_leaks: $claude_leaks,
            claude_note: $claude_note,
            error: $error
        }' >> "$summary_jsonl"
}

process_one() {
    local idx="$1"
    local f="$2"
    local pages="$3"
    local doc_type="$4"

    local base safe prefix
    local out_json out_txt out_claude_raw out_claude_verdict out_claude_err
    local body_tmp http_line http_status duration_s attempt
    local ocr_used confidence input_characters processed_characters remaining_characters
    local claude_result claude_leaks claude_note error_msg
    local claude_exit result_line leaks_line note_line

    base="$(basename "$f")"
    safe="$(safe_name "$base")"
    prefix="$(printf "%03d__%s" "$idx" "$safe")"

    out_json="${run_dir}/${prefix}.response.json"
    out_txt="${run_dir}/${prefix}.anonymized.txt"
    out_claude_raw="${run_dir}/${prefix}.claude.raw.txt"
    out_claude_verdict="${run_dir}/${prefix}.claude.verdict.json"
    out_claude_err="${run_dir}/${prefix}.claude.stderr.txt"

    body_tmp="$(mktemp)"
    http_status="000"
    duration_s="0"
    for attempt in 1 2; do
        http_line="$({
            curl -sS -o "$body_tmp" -w "%{http_code} %{time_total}" \
                -X POST "http://127.0.0.1:8000/anonymize-file" \
                -H "Authorization: Bearer ${TOKEN}" \
                -F "document_type=${doc_type}" \
                -F "file=@${f};type=application/pdf";
        } || true)"
        http_status="$(printf "%s" "$http_line" | awk '{print $1}')"
        duration_s="$(printf "%s" "$http_line" | awk '{print $2}')"
        if [ "$http_status" = "401" ] && [ "$attempt" -eq 1 ]; then
            TOKEN="$(mint_token)"
            continue
        fi
        break
    done

    cp "$body_tmp" "$out_json"
    rm -f "$body_tmp"

    ocr_used=""
    confidence=""
    input_characters=""
    processed_characters=""
    remaining_characters=""
    claude_result="SKIP"
    claude_leaks=""
    claude_note=""
    error_msg=""

    if [ "$http_status" != "200" ]; then
        error_msg="http_${http_status}"
        echo "$(csv_escape "$idx"),$(csv_escape "$base"),$(csv_escape "$pages"),$(csv_escape "$doc_type"),$(csv_escape "$http_status"),$(csv_escape "$duration_s"),$(csv_escape "$ocr_used"),$(csv_escape "$confidence"),$(csv_escape "$input_characters"),$(csv_escape "$processed_characters"),$(csv_escape "$remaining_characters"),$(csv_escape "$claude_result"),$(csv_escape "$claude_leaks"),$(csv_escape "$claude_note"),$(csv_escape "$error_msg")" >> "$summary_csv"
        jsonl_write "$idx" "$f" "$pages" "$doc_type" "$http_status" "$duration_s" "false" "0" "0" "0" "0" "$claude_result" "$claude_leaks" "$claude_note" "$error_msg"
        echo "$f" >> "$processed_file"
        return
    fi

    jq -r '.anonymized_text // ""' < "$out_json" > "$out_txt"
    ocr_used="$(jq -r '.ocr_used // false' < "$out_json")"
    confidence="$(jq -r '.confidence // 0' < "$out_json")"
    input_characters="$(jq -r '.input_characters // 0' < "$out_json")"
    processed_characters="$(jq -r '.processed_characters // 0' < "$out_json")"
    remaining_characters="$(jq -r '.remaining_characters // 0' < "$out_json")"

    claude_exit=0
    (
        cat <<'PROMPT'
Du prüfst die Qualität einer ANONYMISIERTEN deutschen Rechtsdokument-Ausgabe.
Aufgabe: Entscheide, ob noch personenbezogene Daten im Klartext vorhanden sind.

Kriterien:
- PASS: Keine klaren Personenbezüge mehr sichtbar (Namen, Geburtsdaten, Geburtsorte, Adressen, Aktenzeichen/IDs).
- FAIL: Mindestens ein solcher Klartextbezug ist noch vorhanden.
- Platzhalter wie [PERSON], [DATUM], [ORT], [ADRESSE], [AKTENZEICHEN], [GEBURTSDATUM], [GEBURTSORT] sind korrekt und KEIN Leak.

Antworte EXAKT in diesen 3 Zeilen (ohne Markdown):
RESULT: PASS oder FAIL
LEAKS: none oder komma-separierte konkrete Fundstellen
NOTE: kurzer deutscher Satz (max. 25 Wörter)

TEXT:
PROMPT
        cat "$out_txt"
    ) | timeout "${claude_timeout_s}s" env -u ANTHROPIC_API_KEY -u CLAUDE_API_KEY claude -p > "$out_claude_raw" 2> "$out_claude_err" || claude_exit=$?

    result_line="$(grep -Eim1 '^RESULT:[[:space:]]*(PASS|FAIL)\b' "$out_claude_raw" || true)"
    leaks_line="$(grep -Eim1 '^LEAKS:' "$out_claude_raw" || true)"
    note_line="$(grep -Eim1 '^NOTE:' "$out_claude_raw" || true)"

    if [ -n "$result_line" ]; then
        claude_result="$(echo "$result_line" | sed -E 's/^RESULT:[[:space:]]*(PASS|FAIL).*/\1/I' | tr '[:lower:]' '[:upper:]')"
        claude_leaks="$(trim_ws "${leaks_line#*:}")"
        claude_note="$(trim_ws "${note_line#*:}")"
        if [ -z "$claude_leaks" ]; then
            claude_leaks="none"
        fi
    elif jq -e . "$out_claude_raw" >/dev/null 2>&1; then
        # JSON fallback if Claude emitted JSON.
        if jq -e '.result' "$out_claude_raw" >/dev/null 2>&1; then
            claude_result="$(jq -r '.result // "ERROR"' "$out_claude_raw" | tr '[:lower:]' '[:upper:]')"
            claude_leaks="$(jq -r '.leaks // "none"' "$out_claude_raw" | tr '\n' ' ')"
            claude_note="$(jq -r '.note // ""' "$out_claude_raw" | tr '\n' ' ')"
        elif jq -e '.result | fromjson | .result' "$out_claude_raw" >/dev/null 2>&1; then
            claude_result="$(jq -r '.result | fromjson | .result // "ERROR"' "$out_claude_raw" | tr '[:lower:]' '[:upper:]')"
            claude_leaks="$(jq -r '.result | fromjson | (.leaks // "none")' "$out_claude_raw" | tr '\n' ' ')"
            claude_note="$(jq -r '.result | fromjson | .note // ""' "$out_claude_raw" | tr '\n' ' ')"
        else
            claude_result="ERROR"
        fi
    else
        claude_result="ERROR"
    fi

    if [ "$claude_result" != "PASS" ] && [ "$claude_result" != "FAIL" ]; then
        claude_result="ERROR"
        if [ "$claude_exit" = "124" ]; then
            error_msg="claude_timeout"
        elif [ "$claude_exit" -ne 0 ]; then
            error_msg="claude_exit_${claude_exit}"
        else
            error_msg="claude_unparseable_output"
        fi
    fi

    jq -n --arg result "$claude_result" --arg leaks "$claude_leaks" --arg note "$claude_note" '{result:$result,leaks:$leaks,note:$note}' > "$out_claude_verdict"

    echo "$(csv_escape "$idx"),$(csv_escape "$base"),$(csv_escape "$pages"),$(csv_escape "$doc_type"),$(csv_escape "$http_status"),$(csv_escape "$duration_s"),$(csv_escape "$ocr_used"),$(csv_escape "$confidence"),$(csv_escape "$input_characters"),$(csv_escape "$processed_characters"),$(csv_escape "$remaining_characters"),$(csv_escape "$claude_result"),$(csv_escape "$claude_leaks"),$(csv_escape "$claude_note"),$(csv_escape "$error_msg")" >> "$summary_csv"
    jsonl_write "$idx" "$f" "$pages" "$doc_type" "$http_status" "$duration_s" "$ocr_used" "$confidence" "$input_characters" "$processed_characters" "$remaining_characters" "$claude_result" "$claude_leaks" "$claude_note" "$error_msg"

    echo "$f" >> "$processed_file"
}

mapfile -d '' all_pdfs < <(
    for d in "${dirs[@]}"; do
        if [ -d "$d" ]; then
            find "$d" -type f -name '*.pdf' -print0
        else
            echo "WARN: missing dir: $d" >&2
        fi
    done | sort -z -u
)

if [ "${#all_pdfs[@]}" -eq 0 ]; then
    echo "No PDFs found in input dirs." >&2
    exit 1
fi

jq -n \
    --arg run_id "$run_id" \
    --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --argjson max_docs "$max_docs" \
    --argjson max_pages "$max_pages" \
    --argjson claude_timeout_s "$claude_timeout_s" \
    --arg dry_run "$dry_run" \
    --arg resume "$resume" \
    --arg forced_type "$forced_type" \
    --argjson dirs "$(printf '%s\n' "${dirs[@]}" | jq -R . | jq -s .)" \
    '{
        run_id: $run_id,
        started_at: $started_at,
        max_docs: $max_docs,
        max_pages: $max_pages,
        claude_timeout_s: $claude_timeout_s,
        dry_run: ($dry_run == "1"),
        resume: ($resume == "1"),
        forced_type: $forced_type,
        dirs: $dirs
    }' > "$meta_json"

if [ "$dry_run" = "0" ]; then
    if ! curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
        echo "App health check failed: http://127.0.0.1:8000/health" >&2
        exit 1
    fi
    TOKEN="$(mint_token)"
fi

selected=0
seen=0
skipped_pages=0
skipped_resume=0
skipped_page_err=0

echo "Run dir: $run_dir"
echo "Max docs: $max_docs | Max pages: $max_pages | Claude timeout: ${claude_timeout_s}s"
echo "Scanning ${#all_pdfs[@]} PDFs..."

for f in "${all_pdfs[@]}"; do
    if [ "$selected" -ge "$max_docs" ]; then
        break
    fi
    seen=$((seen + 1))

    if [ "$resume" = "1" ] && grep -Fxq "$f" "$processed_file"; then
        skipped_resume=$((skipped_resume + 1))
        continue
    fi

    pages="$(page_count "$f" || true)"
    if ! [[ "${pages:-}" =~ ^[0-9]+$ ]]; then
        skipped_page_err=$((skipped_page_err + 1))
        echo "$(csv_escape "$f"),$(csv_escape ""),$(csv_escape "page_count_error")" >> "$skipped_csv"
        continue
    fi
    if [ "$pages" -gt "$max_pages" ]; then
        skipped_pages=$((skipped_pages + 1))
        echo "$(csv_escape "$f"),$(csv_escape "$pages"),$(csv_escape "page_limit_exceeded")" >> "$skipped_csv"
        continue
    fi

    selected=$((selected + 1))
    idx="$selected"
    doc_type="$forced_type"
    if [ -z "$doc_type" ]; then
        doc_type="$(guess_type "$f")"
    fi

    echo "[$idx/$max_docs] $f (pages=$pages, type=$doc_type)"

    if [ "$dry_run" = "1" ]; then
        echo "$(csv_escape "$f"),$(csv_escape "$pages"),$(csv_escape "selected_dry_run")" >> "$skipped_csv"
        continue
    fi

    process_one "$idx" "$f" "$pages" "$doc_type"
done

echo ""
echo "Done."
echo "Seen: $seen"
echo "Selected (<=${max_pages} pages): $selected"
echo "Skipped >${max_pages} pages: $skipped_pages"
echo "Skipped (page count error): $skipped_page_err"
echo "Skipped (already processed in resume): $skipped_resume"
echo "Artifacts:"
echo "  - $run_dir"
echo "  - $summary_csv"
echo "  - $summary_jsonl"
echo "  - $skipped_csv"
