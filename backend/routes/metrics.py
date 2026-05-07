import pandas as pd
from flask import Blueprint, jsonify
from src import config as cfg

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        adv_path = cfg.METRICS_DIR / "adversarial_results.csv"
        def_path = cfg.METRICS_DIR / "defense_results.csv"
        
        adv_data = []
        if adv_path.exists():
            adv_data = pd.read_csv(adv_path).to_dict(orient="records")
            
        def_data = []
        if def_path.exists():
            def_data = pd.read_csv(def_path).to_dict(orient="records")
            
        return jsonify({
            "adversarial": adv_data,
            "defense": def_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
