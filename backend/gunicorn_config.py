# Gunicorn configuration file for Render deployment
import os

# Bind to the port Render provides
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker configuration
workers = 1  # Reduced for memory efficiency with TensorFlow
worker_class = "sync"
timeout = 120  # Increased timeout for TensorFlow model loading
keepalive = 5

# Preload app to load model once
preload_app = True

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Reduce memory usage
max_requests = 100
max_requests_jitter = 10
