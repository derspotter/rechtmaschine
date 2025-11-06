#!/bin/bash

# Test script for the service manager

echo "=========================================="
echo "Service Manager Test"
echo "=========================================="
echo ""

# Files
OCR_FILE="/home/jayjag/Nextcloud/Kanzlei/test_files/anlage_k4_bescheid.pdf"
MANAGER_URL="http://localhost:8004"

echo "Step 1: Clean slate - kill all services"
echo "----------------------------------------"
pkill -f ocr_service.py
pkill -f anonymization_service.py
pkill -f service_manager.py
sleep 2

echo "VRAM (should be low):"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

echo "Step 2: Start Service Manager"
echo "----------------------------------------"
cd /home/jayjag/Nextcloud/Kanzlei
python3 service_manager.py > /tmp/manager.log 2>&1 &
MANAGER_PID=$!
echo "Manager PID: $MANAGER_PID"
sleep 3

# Check if manager started
if ! curl -s http://localhost:8004/health > /dev/null 2>&1; then
    echo "ERROR: Service manager failed to start"
    echo "Check logs: tail /tmp/manager.log"
    exit 1
fi

echo "Manager is running!"
echo ""

echo "Step 3: Test OCR Request"
echo "----------------------------------------"
echo "Sending OCR request (manager should auto-start OCR service)..."
START=$(date +%s)

curl -s -X POST $MANAGER_URL/ocr \
  -F "file=@$OCR_FILE" \
  > /tmp/manager_ocr_response.json

END=$(date +%s)
DURATION=$((END - START))

echo "OCR completed in ${DURATION}s"
echo ""

# Show OCR result
python3 << 'EOF'
import json
try:
    with open('/tmp/manager_ocr_response.json', 'r') as f:
        data = json.load(f)
    print(f"  Pages: {data.get('page_count')}")
    print(f"  Text length: {len(data.get('full_text', ''))}")
    print(f"  Confidence: {data.get('avg_confidence')}")
except Exception as e:
    print(f"  Error: {e}")
EOF
echo ""

echo "VRAM with OCR loaded:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

echo "Step 4: Check Service Status"
echo "----------------------------------------"
curl -s $MANAGER_URL/status | python3 -m json.tool
echo ""

echo "Step 5: Test Anonymization Request"
echo "----------------------------------------"
echo "Sending anonymization request (manager should switch to anon)..."
START=$(date +%s)

# Create test anonymization request
cat > /tmp/manager_anon_request.json << 'JSON_EOF'
{
  "text": "Anhörung vom 15.03.2024\n\nAnwesend: Herr Max Mustermann (Antragsteller), Frau Sarah Schmidt (Dolmetscherin), Herr Dr. Klaus Weber (Entscheider BAMF)\n\nDer Antragsteller Max Mustermann, geboren am 01.01.1990 in Damaskus, Syrien, gibt zu Protokoll:\n\n\"Ich bin zusammen mit meiner Frau Anna Mustermann und unseren zwei Kindern, Lisa und Tom Mustermann, aus Syrien geflohen. Wir haben Angst vor Verfolgung durch die Assad-Regierung.\"\n\nRichter: Dr. Müller\nProtokollführer: Schmidt\nOrt: München",
  "document_type": "Anhörung"
}
JSON_EOF

curl -s -X POST $MANAGER_URL/anonymize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: GR5Yu7rY1mbOM6HHFA0neboigIl32VbYvfWB3Yl0" \
  -d @/tmp/manager_anon_request.json \
  > /tmp/manager_anon_response.json

END=$(date +%s)
DURATION=$((END - START))

echo "Anonymization completed in ${DURATION}s"
echo ""

# Show anonymization result
python3 << 'EOF'
import json
try:
    with open('/tmp/manager_anon_response.json', 'r') as f:
        data = json.load(f)
    print(f"  Plaintiff names: {data.get('plaintiff_names')}")
    print(f"  Confidence: {data.get('confidence')}")
    print(f"  Text length: {len(data.get('anonymized_text', ''))}")
except Exception as e:
    print(f"  Error: {e}")
EOF
echo ""

echo "VRAM with Anon loaded:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
echo ""

echo "Step 6: Final Status Check"
echo "----------------------------------------"
curl -s $MANAGER_URL/status | python3 -m json.tool
echo ""

echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""

echo "Manager logs:"
tail -20 /tmp/manager.log
echo ""

echo "To stop manager: kill $MANAGER_PID"
echo "To view full logs: tail -f /tmp/manager.log"
echo ""
