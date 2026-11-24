"""
Gunicorn configuration for Crop Health Diagnostic Service v2 (FastAPI / ASGI).
Automatically switches Gunicorn from the default sync WSGI worker to UvicornWorker
so commands like `gunicorn --workers 1 --timeout 300 --preload app:app` work cleanly.
"""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300
keepalive = 5
