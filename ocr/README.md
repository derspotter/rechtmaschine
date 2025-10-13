# Rechtmaschine OCR Service

OCR (Optical Character Recognition) service for scanned PDF documents using PaddleOCR with GPU acceleration.

## Features
- GPU-accelerated OCR with PaddlePaddle
- Supports PDF, PNG, JPG, JPEG, BMP, TIFF
- Multi-page PDF support
- Text orientation detection
- REST API with FastAPI
- Optional API key authentication

## Quick Start

### Start the Service
```bash
cd ~/Nextcloud/Kanzlei/ocr
source venv/bin/activate
python ocr_service.py
```

You should see:
```
============================================================
Rechtmaschine OCR Service
============================================================
PaddlePaddle Version: 3.2.0
GPU Available: True
API Key Auth: Disabled
Listening on: 0.0.0.0:8003
============================================================
```

### Test the Service

**Health check:**
```bash
curl http://localhost:8003/health
```

**OCR a document:**
```bash
curl -X POST http://localhost:8003/ocr \
  -F "file=@/path/to/document.pdf" \
  | jq .
```

**Via Tailscale (from another machine):**
```bash
curl http://100.106.140.46:8003/health
```

## Configuration

### Environment Variables

Create `.env` file (optional):
```bash
# Optional: Set API key for authentication
OCR_API_KEY=your_random_api_key_here

# Optional: Change port (default: 8003)
OCR_PORT=8003
```

### With API Key

If you set `OCR_API_KEY`, include it in requests:
```bash
curl -X POST http://localhost:8003/ocr \
  -H "X-API-Key: your_random_api_key_here" \
  -F "file=@document.pdf"
```

## API Endpoints

### POST /ocr
Perform OCR on uploaded file.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (PDF or image)
- Header (optional): `X-API-Key`

**Response:**
```json
{
  "text": "Extracted text from document...",
  "confidence": 0.95,
  "page_count": 3,
  "language": "en"
}
```

### GET /health
Check service status.

**Response:**
```json
{
  "status": "healthy",
  "paddle_version": "3.2.0",
  "gpu_available": true,
  "gpu_info": "NVIDIA GeForce RTX 3060, 12288 MiB",
  "ocr_engine": "PaddleOCR"
}
```

## Server Integration

Update server `.env` file:
```bash
# OCR Service (via Tailscale to home PC)
OCR_SERVICE_URL=http://100.106.140.46:8003
OCR_API_KEY=matching_key_from_home_pc
```

## Running as Background Service

### Option 1: nohup
```bash
cd ~/Nextcloud/Kanzlei/ocr
source venv/bin/activate
nohup python ocr_service.py > ocr.log 2>&1 &
```

### Option 2: systemd (recommended)

Create `/etc/systemd/system/rechtmaschine-ocr.service`:
```ini
[Unit]
Description=Rechtmaschine OCR Service
After=network.target

[Service]
Type=simple
User=jayjag
WorkingDirectory=/home/jayjag/Nextcloud/Kanzlei/ocr
Environment="PATH=/home/jayjag/Nextcloud/Kanzlei/ocr/venv/bin"
ExecStart=/home/jayjag/Nextcloud/Kanzlei/ocr/venv/bin/python ocr_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rechtmaschine-ocr
sudo systemctl start rechtmaschine-ocr
sudo systemctl status rechtmaschine-ocr
```

## Supported Languages

PaddleOCR supports multiple languages. To change language, modify `ocr_service.py`:

```python
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='german',  # Change to 'german', 'french', 'ch' (Chinese), etc.
    use_gpu=True,
    show_log=False
)
```

Available languages: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md

## Performance

- **First request:** 5-10 seconds (model loading)
- **Subsequent requests:** 1-3 seconds per page
- **GPU VRAM usage:** ~2-3 GB
- **Concurrent requests:** Limited by GPU memory

## Troubleshooting

### Service won't start
Check if port 8003 is in use:
```bash
sudo lsof -i :8003
```

### Poor OCR quality
- Ensure document has good resolution (300+ DPI recommended)
- Try different language models
- Check if document is upside down (angle detection should handle this)

### GPU not being used
Verify GPU availability:
```bash
python -c "import paddle; print('GPU:', paddle.device.is_compiled_with_cuda())"
nvidia-smi
```

### Service logs
View logs if running as systemd service:
```bash
sudo journalctl -u rechtmaschine-ocr -f
```
