#!/bin/bash

# Test script to run OCR and Anonymization simultaneously
# Tests VRAM usage when both models are loaded

echo "=========================================="
echo "Concurrent Service Test"
echo "=========================================="
echo ""

# File paths
OCR_FILE1="/home/jayjag/Nextcloud/Kanzlei/test_files/anlage_k4_bescheid.pdf"
OCR_FILE2="/home/jayjag/Nextcloud/Kanzlei/test_files/20250211_143922_Aufhebungsbescheid.pdf"
ANON_API_KEY="GR5Yu7rY1mbOM6HHFA0neboigIl32VbYvfWB3Yl0"

# Get initial VRAM usage
echo "Initial VRAM usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
echo ""

# Start monitoring VRAM in background
echo "Starting VRAM monitor (will run for 120 seconds)..."
(
    for i in {1..120}; do
        timestamp=$(date +"%H:%M:%S")
        vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        echo "$timestamp - VRAM: ${vram} MiB"
        sleep 1
    done
) > /tmp/vram_monitor.log 2>&1 &
MONITOR_PID=$!

sleep 2

echo ""
echo "=========================================="
echo "Sending requests simultaneously..."
echo "=========================================="
echo ""

# Send OCR request 1 in background
echo "[1] Sending OCR request (anlage_k4_bescheid.pdf - 2.8MB, 24 pages)..."
(
    START=$(date +%s)
    curl -s -X POST http://localhost:8003/ocr \
      -F "file=@$OCR_FILE1" \
      > /tmp/ocr1_response.json 2>&1
    END=$(date +%s)
    DURATION=$((END - START))
    echo "[1] OCR 1 completed in ${DURATION}s"
) &
OCR1_PID=$!

# Extract text directly from machine-readable PDF and anonymize
echo "[2] Extracting text from 20250211_143922_Aufhebungsbescheid.pdf (machine-readable) and anonymizing..."
(
    START=$(date +%s)

    # Extract text directly using PyMuPDF (fitz)
    python3 << 'PYTHON_EOF'
import fitz  # PyMuPDF
import json
import sys

try:
    # Extract text from machine-readable PDF
    doc = fitz.open('/home/jayjag/Nextcloud/Kanzlei/test_files/20250211_143922_Aufhebungsbescheid.pdf')
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()

    print(f"[2] Extracted {len(text)} characters from PDF")

    # Create anonymization request
    anon_request = {
        "text": text,
        "document_type": "Bescheid"
    }

    with open('/tmp/anon_request_bescheid.json', 'w') as f:
        json.dump(anon_request, f, ensure_ascii=False)

except Exception as e:
    print(f"[2] Error extracting text: {e}")
    sys.exit(1)
PYTHON_EOF

    # Send to anonymization service
    curl -s -X POST http://localhost:8002/anonymize \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $ANON_API_KEY" \
      -d @/tmp/anon_request_bescheid.json \
      > /tmp/anon_response.json 2>&1

    END=$(date +%s)
    DURATION=$((END - START))
    echo "[2] Text extraction + Anonymization completed in ${DURATION}s"
) &
ANON_PID=$!

echo ""
echo "Waiting for both requests to complete..."
echo "(This may take 60-120 seconds for first-time model loading)"
echo ""

# Wait for both requests
wait $OCR1_PID
wait $ANON_PID

# Stop monitor after a bit more time
sleep 5
kill $MONITOR_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""

# Show VRAM usage timeline
echo "VRAM usage timeline (last 30 seconds):"
tail -30 /tmp/vram_monitor.log
echo ""

# Find peak VRAM
PEAK=$(awk '{print $5}' /tmp/vram_monitor.log | grep -o '[0-9]*' | sort -n | tail -1)
echo "Peak VRAM usage: ${PEAK} MiB / 12288 MiB"
PERCENT=$(echo "scale=1; $PEAK * 100 / 12288" | bc)
echo "Peak usage: ${PERCENT}%"
echo ""

# Check if we hit limits
if [ $PEAK -gt 11500 ]; then
    echo "⚠️  WARNING: VRAM usage very high (>94%)!"
    echo "   Consider using smaller models or queue-based approach."
elif [ $PEAK -gt 10000 ]; then
    echo "⚠️  CAUTION: VRAM usage moderately high (>81%)."
    echo "   Monitor for potential OOM errors."
else
    echo "✓ VRAM usage within safe limits."
fi
echo ""

# Show response summaries
echo "----------------------------------------"
echo "OCR Response (anlage_k4_bescheid.pdf):"
if [ -f /tmp/ocr1_response.json ]; then
    python3 -c "import json; data=json.load(open('/tmp/ocr1_response.json')); print('Filename:', data.get('filename', 'N/A')); print('Pages:', data.get('page_count', 'N/A')); print('Text length:', len(data.get('full_text', ''))); print('Avg Confidence:', data.get('avg_confidence', 'N/A')); print('First 300 chars:', data.get('full_text', '')[:300])" 2>/dev/null || cat /tmp/ocr1_response.json | head -20
else
    echo "Error: No OCR response file"
fi
echo ""

echo "----------------------------------------"
echo "Anonymization Response (20250211_143922_Aufhebungsbescheid.pdf):"
if [ -f /tmp/anon_response.json ]; then
    python3 -c "import json; data=json.load(open('/tmp/anon_response.json')); print('Plaintiff names:', data.get('plaintiff_names', [])); print('Confidence:', data.get('confidence', 'N/A')); print('Anonymized text (first 500 chars):', data.get('anonymized_text', '')[:500])" 2>/dev/null || cat /tmp/anon_response.json | head -20
else
    echo "Error: No anonymization response file"
fi
echo ""

# Final VRAM check
echo "----------------------------------------"
echo "Final VRAM usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
echo ""

echo "Full logs saved to:"
echo "  - VRAM timeline: /tmp/vram_monitor.log"
echo "  - OCR response: /tmp/ocr1_response.json"
echo "  - Anon request: /tmp/anon_request_bescheid.json"
echo "  - Anon response: /tmp/anon_response.json"
echo ""
