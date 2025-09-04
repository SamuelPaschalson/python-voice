#!/bin/bash

# Memory optimization environment variables
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMBA_NUM_THREADS=2
export MALLOC_TRIM_THRESHOLD_=100000

# Python optimization
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Memory monitoring
export VOICE_PROCESSOR_PORT=5001

echo "Starting voice processor with memory optimizations..."

# Check available memory
echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"

# Start with gunicorn
exec gunicorn \
    --config gunicorn.conf.py \
    --log-level info \
    app:app
