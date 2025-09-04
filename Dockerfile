# Dockerfile.voice-processor
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set memory optimization environment variables
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2
ENV NUMBA_NUM_THREADS=2
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Resemblyzer models to reduce startup memory spike
RUN python -c "from resemblyzer import VoiceEncoder; VoiceEncoder()" && \
    rm -rf /tmp/*

# Copy application code
COPY optimized_voice_processor.py .
COPY gunicorn.conf.py .
COPY start_voice_processor.sh .

# Make script executable
RUN chmod +x start_voice_processor.sh

# Create temp directory in shared memory
RUN mkdir -p /dev/shm/voice_temp

# Set resource limits
LABEL memory_limit="2g"
LABEL cpu_limit="2"

EXPOSE 5001

# Health check
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["./start_voice_processor.sh"]
