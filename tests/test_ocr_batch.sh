#!/bin/bash

set -euo pipefail

API_URL="${OCR_API_URL:-http://localhost:8004/ocr}"
INPUT_DIR="${OCR_INPUT_DIR:-test_files/kanzlei_samples}"
OUT_DIR="${OCR_OUT_DIR:-test_files/ocred/ocr_batch}"
FORMAT="${OCR_OUT_FORMAT:-md}"
PATTERN="*Bescheid*.pdf"
COUNT=""
FILES=()

usage() {
    echo "Usage: $0 [-n count] [-d input_dir] [-p pattern] [-o out_dir] [-f md|txt] [files...]"
    echo ""
    echo "Defaults:"
    echo "  input_dir: $INPUT_DIR"
    echo "  pattern:   $PATTERN"
    echo "  out_dir:   $OUT_DIR"
    echo "  format:    $FORMAT"
    echo ""
    echo "Examples:"
    echo "  $0 -n 1"
    echo "  $0 -n 5 -p '*Bescheid*.pdf'"
    echo "  $0 -f txt -o test_files/ocred/ocr_batch_txt"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--count)
            COUNT="$2"
            shift 2
            ;;
        -d|--dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -p|--pattern)
            PATTERN="$2"
            shift 2
            ;;
        -o|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

if [[ "$FORMAT" != "md" && "$FORMAT" != "txt" ]]; then
    echo "Invalid format: $FORMAT (use md or txt)" >&2
    exit 1
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    mapfile -t FILES < <(rg --files -g "$PATTERN" "$INPUT_DIR" | sort)
fi

if [[ -n "$COUNT" ]]; then
    FILES=( "${FILES[@]:0:$COUNT}" )
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "No files found (dir=$INPUT_DIR pattern=$PATTERN)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "OCR API:  $API_URL"
echo "Output:   $OUT_DIR"
echo "Format:   $FORMAT"
echo "Files:    ${#FILES[@]}"
echo ""

for file in "${FILES[@]}"; do
    echo "Processing: $file"
    tmp_json="$(mktemp /tmp/ocr_batch_XXXX.json)"
    curl -s -F "file=@$file" "$API_URL" > "$tmp_json"

    python3 - "$tmp_json" "$OUT_DIR" "$file" "$FORMAT" << 'PY'
import json
import re
import sys
import unicodedata
from pathlib import Path

json_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
input_file = Path(sys.argv[3])
fmt = sys.argv[4]

def slugify(value: str) -> str:
    value = value.strip()
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "document"

payload = json.loads(json_path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and payload.get("detail"):
    raise SystemExit(f"[ERROR] {input_file.name}: {payload['detail']}")

full_text = ""
if isinstance(payload, dict):
    full_text = payload.get("full_text") or ""

slug = slugify(input_file.stem)
out_path = out_dir / f"ocr_{slug}.{fmt}"
out_path.write_text(full_text, encoding="utf-8")

page_count = payload.get("page_count") if isinstance(payload, dict) else None
avg_conf = payload.get("avg_confidence") if isinstance(payload, dict) else None
line_count = full_text.count("\n") + (1 if full_text else 0)
print(
    f"  -> {out_path} ({len(full_text)} chars, {line_count} lines, pages={page_count}, avg_conf={avg_conf})"
)
PY

    rm -f "$tmp_json"
    echo ""
done
