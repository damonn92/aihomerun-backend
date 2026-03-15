# ── Base image: slim Python 3.11 on Debian Bookworm ──────────────────────────
FROM python:3.11-slim-bookworm

# System deps required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MediaPipe model files so they don't need to be fetched at runtime.
# Also grant write access to the mediapipe modules dir for the non-root user.
RUN python -c "import mediapipe as mp; mp.solutions.pose.Pose()" 2>/dev/null || true
RUN chmod -R a+rw /usr/local/lib/python3.11/site-packages/mediapipe/modules/ 2>/dev/null || true

# Copy source
COPY . .

# Create uploads dir and non-root user
RUN mkdir -p uploads && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Default: API server. Override CMD for worker mode.
# API:    docker run <image>
# Worker: docker run <image> python worker.py
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
