import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

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

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
