FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for Playwright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    openssh-client \
    libnss3 \
    libnspr4 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libcups2t64 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2t64 \
    libatspi2.0-0t64 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache -r requirements.txt

# Install Playwright browsers (skip system deps due to Debian Trixie compatibility)
RUN playwright install chromium

# Copy minimal application stub (bind mount will overlay full source at runtime)
COPY app/data ./data

# Expose port
EXPOSE 8000

# Set Python to run in unbuffered mode for immediate log output
ENV PYTHONUNBUFFERED=1

# Ensure our project root takes precedence on module resolution
ENV PYTHONPATH=/app

# Run the application with hot reload (reload dir must match the bind mount)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info", "--reload", "--reload-dir", "/app", "--lifespan", "auto", "--timeout-graceful-shutdown", "2"]
