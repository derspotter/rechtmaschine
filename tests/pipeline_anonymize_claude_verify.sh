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
  --api-timeout N     Timeout in seconds for each /anonymize-file call. Default: 60
  --claude-timeout N  Timeout in seconds for each Claude verification. Default: 45
  --claude-jobs N     Parallel Claude verifications. Default: 3
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
api_timeout_s=60
claude_timeout_s=45
claude_jobs=3
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
        --api-timeout)
            api_timeout_s="${2:-}"
            shift 2
            ;;
        --claude-timeout)
            claude_timeout_s="${2:-}"
            shift 2
            ;;
        --claude-jobs)
            claude_jobs="${2:-}"
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
if ! [[ "$api_timeout_s" =~ ^[0-9]+$ ]] || [ "$api_timeout_s" -le 0 ]; then
    echo "--api-timeout must be a positive integer" >&2
    exit 2
fi
if ! [[ "$claude_timeout_s" =~ ^[0-9]+$ ]] || [ "$claude_timeout_s" -le 0 ]; then
    echo "--claude-timeout must be a positive integer" >&2
    exit 2
fi
if ! [[ "$claude_jobs" =~ ^[0-9]+$ ]] || [ "$claude_jobs" -le 0 ]; then
    echo "--claude-jobs must be a positive integer" >&2
    exit 2
fi

if [ -z "$run_id" ]; then
    run_id="$(date -u +%Y%m%d-%H%M%S)"
fi

run_dir="tmp/anonymize_pipeline/${run_id}"
processed_file="${run_dir}/processed_paths.txt"
summary_csv="${run_dir}/summary.csv"
summary_jsonl="${run_dir}/summary.jsonl"
summary_lock="${run_dir}/summary.lock"
skipped_csv="${run_dir}/skipped.csv"
meta_json="${run_dir}/run_meta.json"
rows_csv_dir="${run_dir}/rows_csv"
rows_jsonl_dir="${run_dir}/rows_jsonl"

if [ -d "$run_dir" ] && [ "$resume" = "0" ]; then
    echo "Run directory already exists: $run_dir" >&2
    echo "Use --resume with the same --run-id to continue." >&2
    exit 2
fi

mkdir -p "$run_dir"
mkdir -p "$rows_csv_dir"
mkdir -p "$rows_jsonl_dir"
touch "$processed_file"

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
    local token_sub="${ANON_TEST_TOKEN_SUB:-admin@example.com}"
    docker exec rechtmaschine-app bash -lc "cd /app && TOKEN_SUB='${token_sub}' python3 - <<'PY'
from datetime import timedelta
import os
from auth import create_access_token
token_sub = os.getenv('TOKEN_SUB', 'admin@example.com')
print(create_access_token({'sub': token_sub}, expires_delta=timedelta(hours=8)))
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
    local extraction_prompt_tokens="${12}"
    local extraction_completion_tokens="${13}"
    local extraction_total_duration_ns="${14}"
    local extraction_model="${15}"
    local extraction_format="${16}"
    local extraction_temperature="${17}"
    local extraction_num_ctx="${18}"
    local extraction_top_k="${19}"
    local extraction_top_p="${20}"
    local extraction_min_p="${21}"
    local extraction_repeat_penalty="${22}"
    local claude_result="${23}"
    local claude_leaks="${24}"
    local claude_note="${25}"
    local error_msg="${26}"

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
        --arg extraction_prompt_tokens "$extraction_prompt_tokens" \
        --arg extraction_completion_tokens "$extraction_completion_tokens" \
        --arg extraction_total_duration_ns "$extraction_total_duration_ns" \
        --arg extraction_model "$extraction_model" \
        --arg extraction_format "$extraction_format" \
        --arg extraction_temperature "$extraction_temperature" \
        --arg extraction_num_ctx "$extraction_num_ctx" \
        --arg extraction_top_k "$extraction_top_k" \
        --arg extraction_top_p "$extraction_top_p" \
        --arg extraction_min_p "$extraction_min_p" \
        --arg extraction_repeat_penalty "$extraction_repeat_penalty" \
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
            extraction_prompt_tokens: ($extraction_prompt_tokens|tonumber),
            extraction_completion_tokens: ($extraction_completion_tokens|tonumber),
            extraction_total_duration_ns: ($extraction_total_duration_ns|tonumber),
            extraction_model: $extraction_model,
            extraction_format: $extraction_format,
            extraction_temperature: ($extraction_temperature|tonumber),
            extraction_num_ctx: ($extraction_num_ctx|tonumber),
            extraction_top_k: ($extraction_top_k|tonumber),
            extraction_top_p: ($extraction_top_p|tonumber),
            extraction_min_p: ($extraction_min_p|tonumber),
            extraction_repeat_penalty: ($extraction_repeat_penalty|tonumber),
            claude_result: $claude_result,
            claude_leaks: $claude_leaks,
            claude_note: $claude_note,
            error: $error
        }'
}

write_row_files() {
    local idx="$1"
    local base="$2"
    local full_path="$3"
    local pages="$4"
    local doc_type="$5"
    local http_status="$6"
    local duration_s="$7"
    local ocr_used="$8"
    local confidence="$9"
    local input_characters="${10}"
    local processed_characters="${11}"
    local remaining_characters="${12}"
    local extraction_prompt_tokens="${13}"
    local extraction_completion_tokens="${14}"
    local extraction_total_duration_ns="${15}"
    local extraction_model="${16}"
    local extraction_format="${17}"
    local extraction_temperature="${18}"
    local extraction_num_ctx="${19}"
    local extraction_top_k="${20}"
    local extraction_top_p="${21}"
    local extraction_min_p="${22}"
    local extraction_repeat_penalty="${23}"
    local claude_result="${24}"
    local claude_leaks="${25}"
    local claude_note="${26}"
    local error_msg="${27}"

    local row_csv row_jsonl
    row_csv="${rows_csv_dir}/$(printf '%06d' "$idx")__$(safe_name "$base").csv"
    row_jsonl="${rows_jsonl_dir}/$(printf '%06d' "$idx")__$(safe_name "$base").jsonl"

    echo "$(csv_escape "$idx"),$(csv_escape "$base"),$(csv_escape "$pages"),$(csv_escape "$doc_type"),$(csv_escape "$http_status"),$(csv_escape "$duration_s"),$(csv_escape "$ocr_used"),$(csv_escape "$confidence"),$(csv_escape "$input_characters"),$(csv_escape "$processed_characters"),$(csv_escape "$remaining_characters"),$(csv_escape "$extraction_prompt_tokens"),$(csv_escape "$extraction_completion_tokens"),$(csv_escape "$extraction_total_duration_ns"),$(csv_escape "$extraction_model"),$(csv_escape "$extraction_format"),$(csv_escape "$extraction_temperature"),$(csv_escape "$extraction_num_ctx"),$(csv_escape "$extraction_top_k"),$(csv_escape "$extraction_top_p"),$(csv_escape "$extraction_min_p"),$(csv_escape "$extraction_repeat_penalty"),$(csv_escape "$claude_result"),$(csv_escape "$claude_leaks"),$(csv_escape "$claude_note"),$(csv_escape "$error_msg")" > "$row_csv"
    jsonl_write "$idx" "$full_path" "$pages" "$doc_type" "$http_status" "$duration_s" "$ocr_used" "$confidence" "$input_characters" "$processed_characters" "$remaining_characters" "$extraction_prompt_tokens" "$extraction_completion_tokens" "$extraction_total_duration_ns" "$extraction_model" "$extraction_format" "$extraction_temperature" "$extraction_num_ctx" "$extraction_top_k" "$extraction_top_p" "$extraction_min_p" "$extraction_repeat_penalty" "$claude_result" "$claude_leaks" "$claude_note" "$error_msg" > "$row_jsonl"

    # Continuously append row outputs so live monitoring works during long runs.
    (
        flock -x 9
        if [ ! -f "$summary_csv" ]; then
            echo "index,file,pages,document_type,http_status,duration_s,ocr_used,confidence,input_characters,processed_characters,remaining_characters,extraction_prompt_tokens,extraction_completion_tokens,extraction_total_duration_ns,extraction_model,extraction_format,extraction_temperature,extraction_num_ctx,extraction_top_k,extraction_top_p,extraction_min_p,extraction_repeat_penalty,claude_result,claude_leaks,claude_note,error" > "$summary_csv"
        fi
        if [ ! -f "$summary_jsonl" ]; then
            : > "$summary_jsonl"
        fi
        cat "$row_csv" >> "$summary_csv"
        cat "$row_jsonl" >> "$summary_jsonl"
    ) 9>>"$summary_lock"
}

verify_claude_and_write_row() {
    local idx="$1"
    local base="$2"
    local full_path="$3"
    local pages="$4"
    local doc_type="$5"
    local http_status="$6"
    local duration_s="$7"
    local ocr_used="$8"
    local confidence="$9"
    local input_characters="${10}"
    local processed_characters="${11}"
    local remaining_characters="${12}"
    local extraction_prompt_tokens="${13}"
    local extraction_completion_tokens="${14}"
    local extraction_total_duration_ns="${15}"
    local extraction_model="${16}"
    local extraction_format="${17}"
    local extraction_temperature="${18}"
    local extraction_num_ctx="${19}"
    local extraction_top_k="${20}"
    local extraction_top_p="${21}"
    local extraction_min_p="${22}"
    local extraction_repeat_penalty="${23}"
    local out_txt="${24}"
    local out_claude_raw="${25}"
    local out_claude_verdict="${26}"
    local out_claude_err="${27}"

    local claude_result claude_leaks claude_note error_msg
    local claude_exit result_line leaks_line note_line

    claude_result="ERROR"
    claude_leaks="none"
    claude_note=""
    error_msg=""
    claude_exit=0

    (
        cat <<'PROMPT'
Prüfe den anonymisierten deutschen Rechtstext auf verbleibende personenbezogene Klartextdaten.
FAIL nur bei: Klarnamen natürlicher Personen, Geburtsdatum/-ort im Personenkontext, privater Anschrift, individueller persönlicher ID (z. B. AZR, Pass, Ausweis).
Kein Leak: Aktenzeichen/ECLI/Zitate, Behörden- und Gerichtsnamen, allgemeine Ortsnennungen ohne Privatadressbezug, Platzhalter wie [PERSON]/[DATUM]/[ORT]/[ID].
Antworte exakt in 3 Zeilen:
RESULT: PASS oder FAIL
LEAKS: none oder kurze komma-separierte Fundstellen
NOTE: max. 15 Wörter
TEXT:
PROMPT
        cat "$out_txt"
    ) | timeout "${claude_timeout_s}s" env -u ANTHROPIC_API_KEY -u CLAUDE_API_KEY claude -p > "$out_claude_raw" 2> "$out_claude_err" || claude_exit=$?

    result_line="$(grep -Ei '^RESULT:[[:space:]]*(PASS|FAIL)\b' "$out_claude_raw" | tail -n 1 || true)"
    leaks_line="$(grep -Ei '^LEAKS:' "$out_claude_raw" | tail -n 1 || true)"
    note_line="$(grep -Ei '^NOTE:' "$out_claude_raw" | tail -n 1 || true)"

    if [ -n "$result_line" ]; then
        claude_result="$(echo "$result_line" | sed -E 's/^RESULT:[[:space:]]*(PASS|FAIL).*/\1/I' | tr '[:lower:]' '[:upper:]')"
        claude_leaks="$(trim_ws "${leaks_line#*:}")"
        claude_note="$(trim_ws "${note_line#*:}")"
        if [ -z "$claude_leaks" ]; then
            claude_leaks="none"
        fi
    elif jq -e . "$out_claude_raw" >/dev/null 2>&1; then
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

    write_row_files "$idx" "$base" "$full_path" "$pages" "$doc_type" "$http_status" "$duration_s" "$ocr_used" "$confidence" "$input_characters" "$processed_characters" "$remaining_characters" "$extraction_prompt_tokens" "$extraction_completion_tokens" "$extraction_total_duration_ns" "$extraction_model" "$extraction_format" "$extraction_temperature" "$extraction_num_ctx" "$extraction_top_k" "$extraction_top_p" "$extraction_min_p" "$extraction_repeat_penalty" "$claude_result" "$claude_leaks" "$claude_note" "$error_msg"
}

running_claude_jobs=0

wait_for_claude_slot() {
    if [ "$claude_jobs" -le 1 ]; then
        return
    fi
    while [ "$running_claude_jobs" -ge "$claude_jobs" ]; do
        if wait -n; then
            :
        fi
        running_claude_jobs=$((running_claude_jobs - 1))
    done
}

wait_for_all_claude_jobs() {
    if [ "$claude_jobs" -le 1 ]; then
        return
    fi
    while [ "$running_claude_jobs" -gt 0 ]; do
        if wait -n; then
            :
        fi
        running_claude_jobs=$((running_claude_jobs - 1))
    done
}

launch_claude_verification() {
    if [ "$claude_jobs" -le 1 ]; then
        verify_claude_and_write_row "$@"
        return
    fi
    wait_for_claude_slot
    verify_claude_and_write_row "$@" &
    running_claude_jobs=$((running_claude_jobs + 1))
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
    local extraction_prompt_tokens extraction_completion_tokens extraction_total_duration_ns
    local extraction_model extraction_format extraction_temperature extraction_num_ctx
    local extraction_top_k extraction_top_p extraction_min_p extraction_repeat_penalty

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
            curl -sS --max-time "${api_timeout_s}" -o "$body_tmp" -w "%{http_code} %{time_total}" \
                -X POST "http://127.0.0.1:8000/anonymize-file" \
                -H "Authorization: Bearer ${TOKEN}" \
                -F "document_type=${doc_type}" \
                -F "file=@${f};type=application/pdf";
        } || true)"
        http_status="$(printf "%s" "$http_line" | awk '{print $1}')"
        duration_s="$(printf "%s" "$http_line" | awk '{print $2}')"
        if [ -z "$http_status" ]; then
            http_status="000"
        fi
        if [ -z "$duration_s" ]; then
            duration_s="0"
        fi
        if [ "$http_status" = "401" ] && [ "$attempt" -eq 1 ]; then
            TOKEN="$(mint_token)"
            continue
        fi
        break
    done

    cp "$body_tmp" "$out_json"
    rm -f "$body_tmp"

    if [ "$http_status" != "200" ]; then
        write_row_files "$idx" "$base" "$f" "$pages" "$doc_type" "$http_status" "$duration_s" "false" "0" "0" "0" "0" "0" "0" "0" "" "" "0" "0" "0" "0" "0" "0" "SKIP" "" "" "http_${http_status}"
        echo "$f" >> "$processed_file"
        return
    fi

    jq -r '.anonymized_text // ""' < "$out_json" > "$out_txt"
    ocr_used="$(jq -r '.ocr_used // false' < "$out_json")"
    confidence="$(jq -r '.confidence // 0' < "$out_json")"
    input_characters="$(jq -r '.input_characters // 0' < "$out_json")"
    processed_characters="$(jq -r '.processed_characters // 0' < "$out_json")"
    remaining_characters="$(jq -r '.remaining_characters // 0' < "$out_json")"
    extraction_prompt_tokens="$(jq -r '.extraction_prompt_tokens // 0' < "$out_json")"
    extraction_completion_tokens="$(jq -r '.extraction_completion_tokens // 0' < "$out_json")"
    extraction_total_duration_ns="$(jq -r '.extraction_total_duration_ns // 0' < "$out_json")"
    extraction_model="$(jq -r '.extraction_inference_params.model // ""' < "$out_json")"
    extraction_format="$(jq -r '.extraction_inference_params.format // ""' < "$out_json")"
    extraction_temperature="$(jq -r '.extraction_inference_params.temperature // 0' < "$out_json")"
    extraction_num_ctx="$(jq -r '.extraction_inference_params.num_ctx // 0' < "$out_json")"
    extraction_top_k="$(jq -r '.extraction_inference_params.top_k // 0' < "$out_json")"
    extraction_top_p="$(jq -r '.extraction_inference_params.top_p // 0' < "$out_json")"
    extraction_min_p="$(jq -r '.extraction_inference_params.min_p // 0' < "$out_json")"
    extraction_repeat_penalty="$(jq -r '.extraction_inference_params.repeat_penalty // 0' < "$out_json")"

    launch_claude_verification "$idx" "$base" "$f" "$pages" "$doc_type" "$http_status" "$duration_s" "$ocr_used" "$confidence" "$input_characters" "$processed_characters" "$remaining_characters" "$extraction_prompt_tokens" "$extraction_completion_tokens" "$extraction_total_duration_ns" "$extraction_model" "$extraction_format" "$extraction_temperature" "$extraction_num_ctx" "$extraction_top_k" "$extraction_top_p" "$extraction_min_p" "$extraction_repeat_penalty" "$out_txt" "$out_claude_raw" "$out_claude_verdict" "$out_claude_err"
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
    --argjson api_timeout_s "$api_timeout_s" \
    --argjson claude_timeout_s "$claude_timeout_s" \
    --argjson claude_jobs "$claude_jobs" \
    --arg dry_run "$dry_run" \
    --arg resume "$resume" \
    --arg forced_type "$forced_type" \
    --argjson dirs "$(printf '%s\n' "${dirs[@]}" | jq -R . | jq -s .)" \
    '{
        run_id: $run_id,
        started_at: $started_at,
        max_docs: $max_docs,
        max_pages: $max_pages,
        api_timeout_s: $api_timeout_s,
        claude_timeout_s: $claude_timeout_s,
        claude_jobs: $claude_jobs,
        dry_run: ($dry_run == "1"),
        resume: ($resume == "1"),
        forced_type: $forced_type,
        dirs: $dirs
    }' > "$meta_json"

if [ "$dry_run" = "0" ]; then
    health_ok=0
    for _try in 1 2 3; do
        if curl -sf --max-time 3 "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
            health_ok=1
            break
        fi
        sleep 1
    done
    if [ "$health_ok" = "0" ]; then
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
echo "Max docs: $max_docs | Max pages: $max_pages | API timeout: ${api_timeout_s}s | Claude timeout: ${claude_timeout_s}s | Claude jobs: ${claude_jobs}"
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

if [ "$dry_run" = "0" ]; then
    wait_for_all_claude_jobs
fi

echo "index,file,pages,document_type,http_status,duration_s,ocr_used,confidence,input_characters,processed_characters,remaining_characters,extraction_prompt_tokens,extraction_completion_tokens,extraction_total_duration_ns,extraction_model,extraction_format,extraction_temperature,extraction_num_ctx,extraction_top_k,extraction_top_p,extraction_min_p,extraction_repeat_penalty,claude_result,claude_leaks,claude_note,error" > "$summary_csv"
: > "$summary_jsonl"

while IFS= read -r row_csv; do
    cat "$row_csv" >> "$summary_csv"
done < <(find "$rows_csv_dir" -type f -name '*.csv' | sort)

while IFS= read -r row_jsonl; do
    cat "$row_jsonl" >> "$summary_jsonl"
done < <(find "$rows_jsonl_dir" -type f -name '*.jsonl' | sort)

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
