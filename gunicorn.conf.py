# gunicorn.conf.py
import multiprocessing
import os

# Worker configuration - CRITICAL for memory management
workers = 1  # Single worker to avoid multiple model copies
worker_class = "sync"  # Use sync workers, not gevent
worker_connections = 1000
timeout = 120  # Increase timeout for voice processing
keepalive = 2

# Memory management
max_requests = 100  # Restart worker after 100 requests
max_requests_jitter = 20  # Add randomness to restarts
preload_app = True  # Load app before forking (saves memory)

# Process limits
worker_tmp_dir = "/dev/shm"  # Use shared memory for temp files
worker_rlimit_as = 2 * 1024 * 1024 * 1024  # 2GB memory limit per worker
worker_rlimit_nofile = 1024

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Environment
raw_env = [
    'OMP_NUM_THREADS=2',
    'MKL_NUM_THREADS=2',
    'OPENBLAS_NUM_THREADS=2',
    'NUMBA_NUM_THREADS=2',
]

def when_ready(server):
    server.log.info("Voice processor server is ready")

def worker_int(worker):
    worker.log.info("Worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
