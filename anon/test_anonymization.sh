#!/bin/bash
# Test anonymization service with sample German legal text

echo "Testing Qwen3 14B Anonymization Service"
echo "========================================"
echo ""
echo "Sending request to http://localhost:8002/anonymize"
echo ""
echo "Text to anonymize:"
echo "  - Sample German asylum hearing protocol"
echo "  - Contains plaintiff names (Max, Anna, Lisa, Tom Mustermann)"
echo "  - Contains official names (Dr. Müller, Schmidt, Dr. Weber)"
echo ""
echo "Starting request (this may take 30-60 seconds for first request)..."
echo ""

# Start monitoring GPU in background
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv --loop=1 > /tmp/gpu_usage.log 2>&1 &
GPU_MON_PID=$!

# Make the request using JSON file
START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST http://localhost:8002/anonymize \
  -H "Content-Type: application/json" \
  -d @test_request.json \
  2>&1)

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop GPU monitoring
kill $GPU_MON_PID 2>/dev/null

echo ""
echo "Request completed in ${DURATION} seconds"
echo ""
echo "Response:"
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
echo ""
echo "GPU Usage Summary (from monitoring):"
tail -5 /tmp/gpu_usage.log
rm -f /tmp/gpu_usage.log

echo ""
echo "Current GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
