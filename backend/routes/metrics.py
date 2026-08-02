import json
from flask import Blueprint, jsonify
from src import config as cfg

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        json_path = cfg.OUTPUTS_DIR / "evaluation" / "comprehensive_evaluation_results.json"
        if not json_path.exists():
            json_path = cfg.METRICS_DIR / "comprehensive_evaluation_results.json"
        if not json_path.exists():
            json_path = cfg.METRICS_DIR / "full_evaluation_results.json"
        
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({
                "status": "success",
                "recommended_model": "RASC-Net Proposed (Exp 3)",
                "data": data,
            }), 200

        return jsonify({"error": "Evaluation results JSON not found"}), 444
    except Exception as e:
        return jsonify({"error": str(e)}), 500
