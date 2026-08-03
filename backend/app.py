import sys
import os
import signal
import threading
import logging
import faulthandler

# Enable Python C-level fault handler for segfaults/aborts
faulthandler.enable()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app_startup")

def handle_signal(sig, frame):
    try:
        sig_name = signal.Signals(sig).name
    except Exception:
        sig_name = str(sig)
    
    logger.critical(
        f"[SIGNAL INTERCEPTOR] Process PID: {os.getpid()} | Thread: {threading.get_ident()} | "
        f"Received Signal {sig} ({sig_name})!"
    )
    faulthandler.dump_traceback()

# Register signal handlers for graceful fault tracing
for sig_name in ("SIGTERM", "SIGINT", "SIGABRT"):
    if hasattr(signal, sig_name):
        try:
            sig = getattr(signal, sig_name)
            signal.signal(sig, handle_signal)
            logger.info(f"[SIGNAL INTERCEPTOR] Registered signal handler for {sig_name}")
        except Exception as e:
            logger.warning(f"[SIGNAL INTERCEPTOR] Failed to register {sig_name}: {e}")

from download_models import download_models

# Execute automatic model check/download before backend startup
download_models()

from flask import Flask
from flask_cors import CORS

from routes.health import health_bp
from routes.predict import predict_bp
from routes.attacks import attacks_bp
from routes.gradcam import gradcam_bp
from routes.metrics import metrics_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(health_bp, url_prefix='/api')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(attacks_bp, url_prefix='/api')
    app.register_blueprint(gradcam_bp, url_prefix='/api')
    app.register_blueprint(metrics_bp, url_prefix='/api')

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
