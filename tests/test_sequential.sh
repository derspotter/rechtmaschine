#!/bin/bash

# Sequential test: OCR first, unload it, then anonymize
# Tests if freeing OCR VRAM improves anonymization speed

echo "=========================================="
echo "Sequential Test (OCR → Unload → Anonymize)"
echo "=========================================="
echo ""

# File paths
OCR_FILE1="/home/jayjag/Nextcloud/Kanzlei/test_files/anlage_k4_bescheid.pdf"
OCR_FILE2="/home/jayjag/Nextcloud/Kanzlei/test_files/20250211_143922_Aufhebungsbescheid.pdf"
ANON_API_KEY="GR5Yu7rY1mbOM6HHFA0neboigIl32VbYvfWB3Yl0"

echo "Step 1: OCR Processing"
echo "----------------------------------------"
echo "Initial VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# Run OCR
echo "Processing: $OCR_FILE1"
START=$(date +%s)
curl -s -X POST http://localhost:8003/ocr \
  -F "file=@$OCR_FILE1" \
  > /tmp/ocr_seq_response.json 2>&1
END=$(date +%s)
DURATION=$((END - START))
echo "OCR completed in ${DURATION}s"
echo ""

echo "VRAM with OCR loaded:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# Unload OCR model
echo "Step 2: Unloading OCR Model"
echo "----------------------------------------"
echo "Killing OCR service to free VRAM..."
pkill -f ocr_service.py
sleep 3

echo "VRAM after unloading OCR:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# Extract text and anonymize
echo "Step 3: Anonymization (with more VRAM available)"
echo "----------------------------------------"

# Extract text
python3 << 'PYTHON_EOF'
import fitz
import json

doc = fitz.open('/home/jayjag/Nextcloud/Kanzlei/test_files/20250211_143922_Aufhebungsbescheid.pdf')
text = ''
for page in doc:
    text += page.get_text()
doc.close()

print(f"Extracted {len(text)} characters")

anon_request = {
    "text": text,
    "document_type": "Bescheid"
}

with open('/tmp/anon_seq_request.json', 'w') as f:
    json.dump(anon_request, f, ensure_ascii=False)
PYTHON_EOF

# Anonymize
echo "Sending to anonymization service..."
START=$(date +%s)
curl -s -X POST http://localhost:8002/anonymize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $ANON_API_KEY" \
  -d @/tmp/anon_seq_request.json \
  > /tmp/anon_seq_response.json 2>&1
END=$(date +%s)
ANON_DURATION=$((END - START))

echo "Anonymization completed in ${ANON_DURATION}s"
echo ""

echo "VRAM during anonymization:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

# Results
echo "=========================================="
echo "Results"
echo "=========================================="
echo ""

echo "Anonymization Response:"
python3 -c "import json; data=json.load(open('/tmp/anon_seq_response.json')); print('Plaintiff names:', data.get('plaintiff_names', [])); print('Confidence:', data.get('confidence', 'N/A')); print('Time taken:', '${ANON_DURATION}s'); print('\nFirst 300 chars:', data.get('anonymized_text', '')[:300])" 2>/dev/null

echo ""
echo "=========================================="
echo "Comparison"
echo "=========================================="
echo "Concurrent (OCR loaded):  152s"
echo "Sequential (OCR unloaded): ${ANON_DURATION}s"

if [ $ANON_DURATION -lt 152 ]; then
    IMPROVEMENT=$((152 - ANON_DURATION))
    PERCENT=$(echo "scale=1; ($IMPROVEMENT * 100) / 152" | bc)
    echo "Improvement: ${IMPROVEMENT}s faster (${PERCENT}%)"
else
    echo "No improvement"
fi
echo ""

echo "Note: Remember to restart OCR service:"
echo "  cd ~/Nextcloud/Kanzlei/ocr && source venv/bin/activate && python ocr_service.py"
echo ""
