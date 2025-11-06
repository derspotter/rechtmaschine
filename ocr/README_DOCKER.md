# PaddleOCR Docker Deployment Guide

Complete Docker setup for both OCR services with GPU acceleration.

## Services Overview

### 1. Optimized Simple OCR (Port 9003)
- **Technology**: PaddleOCR with TensorRT + FP16
- **Speed**: ~0.86s per page (possibly faster with TensorRT)
- **Output**: Raw text lines
- **Use for**: Fast text extraction, search indexing

### 2. PaddleOCR-VL with vLLM (Port 8118 server, 9005 client)
- **Technology**: Vision-Language model with vLLM acceleration
- **Speed**: ~6.84s per page
- **Output**: Structured semantic blocks (headers, paragraphs, titles)
- **Use for**: Document structure understanding, legal analysis

---

## Quick Start - Both Services

```bash
cd /home/jayjag/rechtmaschine/ocr

# Build and start both services
docker compose -f docker-compose-full.yml up -d --build

# Check status
docker compose -f docker-compose-full.yml ps

# View logs
docker compose -f docker-compose-full.yml logs -f
```

**Services will be available at:**
- Simple OCR: `http://localhost:9003/ocr`
- VL vLLM server: `http://localhost:8118` (internal)

---

## Individual Services

### Option 1: Simple OCR Only

```bash
docker compose -f docker-compose-full.yml up -d paddleocr-optimized
```

**Test:**
```bash
curl -F "file=@test.pdf" http://localhost:9003/ocr | jq .
```

### Option 2: VL OCR Only

```bash
docker compose -f docker-compose-full.yml up -d paddleocr-vllm

# Then start Python client service (port 9005)
cd /home/jayjag/rechtmaschine/ocr
source venv_vl/bin/activate
python ocr_vl_service_vllm.py
```

**Test:**
```bash
curl -F "file=@test.pdf" http://localhost:9005/ocr | jq .
```

---

## Performance Expectations

### Optimized Simple OCR (TensorRT)
- **Current (no TensorRT)**: 0.86s/page
- **Expected with TensorRT**: 0.3-0.5s/page (2-3x faster)
- **24-page document**: ~7-12 seconds (vs 21s currently)

### PaddleOCR-VL + vLLM
- **Performance**: 6.84s/page (already optimized)
- **24-page document**: ~2.7 minutes

---

## Configuration

### Simple OCR Optimizations

Edit `ocr_service_optimized.py` to tune:

```python
OCR_ENGINE = PaddleOCR(
    use_tensorrt=True,           # TensorRT acceleration
    precision='fp16',             # FP16 for speed
    rec_batch_num=8,              # Batch size (4-16)
    det_db_box_thresh=0.5,        # Lower = more sensitive
    det_db_unclip_ratio=1.5,      # Box expansion
)
```

### VL vLLM Optimizations

Edit `vllm_config.yaml`:

```yaml
# GPU memory usage (0.3-0.8)
gpu-memory-utilization: 0.8

# Max concurrent sequences (16-256)
max-num-seqs: 128
```

---

## Resource Usage

### GPU Memory (RTX 3060 12GB)

**Running Both Services:**
- Simple OCR: ~2-3 GB
- VL vLLM: ~9-10 GB
- **Total**: ~11-13 GB (may exceed 12GB limit)

**Recommendation**: Run services **one at a time** or use service_manager to auto-switch.

**Option A - Manual switching:**
```bash
# Stop VL, start Simple
docker compose -f docker-compose-full.yml stop paddleocr-vllm
docker compose -f docker-compose-full.yml start paddleocr-optimized

# Stop Simple, start VL
docker compose -f docker-compose-full.yml stop paddleocr-optimized
docker compose -f docker-compose-full.yml start paddleocr-vllm
```

**Option B - Use service_manager.py** (recommended):
The service manager automatically loads/unloads services as needed.

---

## Building the Images

### Build Simple OCR Image

```bash
docker build -f Dockerfile.ocr-optimized -t paddleocr-optimized:latest .
```

### Pull VL vLLM Image

```bash
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest
```

---

## Integration with Rechtmaschine

### Update environment variables:

```bash
# In /var/opt/docker/rechtmaschine/app/.env
OCR_SERVICE_URL=http://localhost:9003          # Simple OCR
OCR_VL_SERVICE_URL=http://localhost:9005        # VL OCR (via Python client)
```

### Usage in app.py:

```python
# For simple text extraction
response = httpx.post(
    "http://localhost:9003/ocr",
    files={"file": pdf_file}
)
text = response.json()["full_text"]

# For structured document analysis
response = httpx.post(
    "http://localhost:9005/ocr",
    files={"file": pdf_file}
)
structured_blocks = response.json()["structured_output"]
```

---

## Monitoring & Logs

```bash
# View all logs
docker compose -f docker-compose-full.yml logs -f

# View specific service
docker compose -f docker-compose-full.yml logs -f paddleocr-optimized
docker compose -f docker-compose-full.yml logs -f paddleocr-vllm

# Check GPU usage
nvidia-smi

# Check service health
curl http://localhost:9003/health
curl http://localhost:8118/health
```

---

## Troubleshooting

### Out of GPU Memory

**Symptom**: CUDA out of memory errors

**Solution**:
```bash
# Stop all services
docker compose -f docker-compose-full.yml down

# Run only one service at a time
docker compose -f docker-compose-full.yml up -d paddleocr-optimized
```

### Service Won't Start

**Check Docker GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

**Rebuild image:**
```bash
docker compose -f docker-compose-full.yml build paddleocr-optimized
docker compose -f docker-compose-full.yml up -d paddleocr-optimized
```

### TensorRT Not Working

**Check logs for TensorRT initialization:**
```bash
docker logs paddleocr-optimized | grep -i tensorrt
```

**If TensorRT fails, service falls back to standard GPU inference (still fast).**

---

## Benchmarking Docker Setup

Once services are running, benchmark them:

```bash
# Benchmark optimized simple OCR
python benchmark_ocr.py "test.pdf" --max-pages 3 --service1 http://localhost:9003

# Compare with VL
python benchmark_ocr.py "test.pdf" --max-pages 3 --service1 http://localhost:9005
```

Expected results:
- Simple OCR (TensorRT): **0.3-0.5s/page** (2-3x faster than current)
- VL + vLLM: **6.84s/page** (already optimized)

---

## Production Deployment

### Systemd Services

Create `/etc/systemd/system/rechtmaschine-ocr.service`:

```ini
[Unit]
Description=Rechtmaschine OCR Services
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/jayjag/rechtmaschine/ocr
ExecStart=/usr/bin/docker compose -f docker-compose-full.yml up -d
ExecStop=/usr/bin/docker compose -f docker-compose-full.yml down
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable rechtmaschine-ocr
sudo systemctl start rechtmaschine-ocr
sudo systemctl status rechtmaschine-ocr
```

### Auto-restart on Failure

Already configured in docker-compose-full.yml:
```yaml
restart: unless-stopped
```

Services will automatically restart if they crash.

---

## Cost-Benefit Analysis

### Current Setup (No Docker)
- Simple OCR: 0.86s/page
- VL OCR: 37.64s/page (no vLLM)

### Docker Setup
- Simple OCR (TensorRT): **0.3-0.5s/page** (2-3x faster)
- VL OCR (vLLM): **6.84s/page** (5.5x faster)

**For 24-page document:**
- Simple OCR: 21s → **7-12s** (saves 9-14 seconds)
- VL OCR: 15 minutes → **2.7 minutes** (saves 12.3 minutes!)

**Worth it?**
- ✅ Absolutely for VL OCR (massive speedup)
- ⚠️ Marginal for simple OCR (already very fast)
