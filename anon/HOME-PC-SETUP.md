# Home PC Setup Guide

This guide explains how to set up the anonymization service on your home PC.

## Prerequisites

- Home PC with GPU (12GB VRAM)
- Tailscale installed and connected
- Python 3.11+
- Linux operating system

## Quick Setup

### 1. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify installation:
```bash
ollama --version
```

### 2. Pull the Qwen3-14B Model

This will download ~8GB:

```bash
ollama pull qwen2.5:14b-instruct-q5_K_M
```

Verify the model is available:
```bash
ollama list
```

You should see `qwen2.5:14b-instruct-q5_K_M` in the list.

### 3. Clone the Repository

```bash
cd ~
git clone <your-repo-url> rechtmaschine
cd rechtmaschine/anon
```

### 4. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure Environment (Optional)

Create a `.env` file if you want to customize settings:

```bash
# Optional: Customize Ollama URL
OLLAMA_URL=http://localhost:11434/api/generate

# Optional: Set API key (must match server)
ANONYMIZATION_API_KEY=your-strong-random-api-key

# Optional: Use different model
OLLAMA_MODEL=qwen2.5:14b-instruct-q5_K_M
```

### 6. Start the Service

```bash
python anonymization_service.py
```

You should see:
```
============================================================
Rechtmaschine Anonymization Service
============================================================
Model: qwen2.5:14b-instruct-q5_K_M
Ollama URL: http://localhost:11434/api/generate
API Key Auth: Disabled
Listening on: 0.0.0.0:8001
============================================================
```

### 7. Test the Service Locally

In another terminal:

```bash
# Test health endpoint
curl http://localhost:8001/health

# Expected output:
# {"status":"healthy","model":"qwen2.5:14b-instruct-q5_K_M",...}
```

## Testing from Server

From the rechtmaschine.de server, test connectivity:

```bash
# Get your home PC Tailscale IP
tailscale ip -4

# Test from server
curl http://<home-pc-tailscale-ip>:8001/health
```

## Running as a Service (Optional)

To keep the service running in the background, you can use systemd:

### Create systemd service file:

```bash
sudo nano /etc/systemd/system/rechtmaschine-anonymization.service
```

Content:
```ini
[Unit]
Description=Rechtmaschine Anonymization Service
After=network.target ollama.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/rechtmaschine/anon
Environment="PATH=/home/your-username/rechtmaschine/anon/venv/bin"
ExecStart=/home/your-username/rechtmaschine/anon/venv/bin/python anonymization_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable rechtmaschine-anonymization
sudo systemctl start rechtmaschine-anonymization
sudo systemctl status rechtmaschine-anonymization
```

View logs:
```bash
sudo journalctl -u rechtmaschine-anonymization -f
```

## Troubleshooting

### Issue: "Model not found"

**Solution:** Pull the model:
```bash
ollama pull qwen2.5:14b-instruct-q5_K_M
```

### Issue: "Connection refused" from server

**Possible causes:**
1. Service not running on home PC
2. Firewall blocking port 8001
3. Service bound to localhost instead of 0.0.0.0

**Solutions:**
```bash
# Check if service is running
ps aux | grep anonymization_service

# Check if port is listening
sudo netstat -tlnp | grep 8001

# Restart service
pkill -f anonymization_service.py
python anonymization_service.py
```

### Issue: "Ollama unreachable"

**Solution:** Start Ollama service:
```bash
sudo systemctl start ollama
sudo systemctl status ollama
```

### Issue: Slow processing (>2 minutes)

**Causes:**
- First request loads model into VRAM (~30s)
- Very long documents
- GPU not being used

**Check GPU usage:**
```bash
nvidia-smi
```

You should see `ollama` using ~10GB VRAM when processing.

### Issue: Out of memory

**Solution:** Use smaller quantization:
```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
```

Update `OLLAMA_MODEL` in your environment.

## Updating

To update the service:

```bash
cd ~/rechtmaschine
git pull
cd anon
source venv/bin/activate
pip install --upgrade -r requirements.txt

# Restart service
pkill -f anonymization_service.py
python anonymization_service.py
```

## Security Notes

- The service listens on `0.0.0.0:8001` (all interfaces)
- Access is restricted to Tailscale network only
- Consider setting `ANONYMIZATION_API_KEY` for additional security
- No public internet exposure (behind Tailscale)

## Performance Expectations

- **First request:** 30-60 seconds (model loading)
- **Subsequent requests:** 30-60 seconds (processing)
- **VRAM usage:** ~10GB
- **RAM usage:** ~4-6GB

## Support

For issues, check:
1. Service logs: `sudo journalctl -u rechtmaschine-anonymization -f`
2. Ollama logs: `sudo journalctl -u ollama -f`
3. GPU status: `nvidia-smi`
4. Tailscale status: `tailscale status`
