import json
import pandas as pd
from flask import Blueprint, jsonify
from src import config as cfg
from evaluate_models import run_full_evaluation

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        json_path = cfg.METRICS_DIR / "full_evaluation_results.json"
        
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
            return jsonify(full_data), 200
            
        # Fallback to CSV files if json does not exist yet
        adv_path = cfg.METRICS_DIR / "adversarial_results.csv"
        def_path = cfg.METRICS_DIR / "defense_results.csv"
        
        adv_data = pd.read_csv(adv_path).to_dict(orient="records") if adv_path.exists() else []
        def_data = pd.read_csv(def_path).to_dict(orient="records") if def_path.exists() else []
        
        return jsonify({
            "adversarial_evaluations": adv_data,
            "defense_evaluations": def_data
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@metrics_bp.route('/metrics/evaluate', methods=['POST'])
def trigger_evaluation():
    try:
        results = run_full_evaluation()
        return jsonify({"message": "Evaluation completed successfully", "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
