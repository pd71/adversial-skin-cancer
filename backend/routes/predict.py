import json
from flask import Blueprint, request, jsonify
from services.inference import predict_ensemble, predict_rasc_net
from services.clinical_risk import calculate_clinical_risk_score

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image file"}), 400

    # Parse metadata (simplified HAM10000 hierarchy inputs)
    metadata_raw = request.form.get('metadata')
    metadata = {}
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
    else:
        metadata = {
            "age_approx": request.form.get("age_approx", 0),
            "sex": request.form.get("sex", "unknown"),
            "anatom_site_1": request.form.get("anatom_site_1", ""),
            "anatom_site_2": request.form.get("anatom_site_2", ""),
            "anatom_site_3": request.form.get("anatom_site_3", ""),
            "melanocytic": request.form.get("melanocytic", "false"),
            "concomitant_biopsy": request.form.get("concomitant_biopsy", "false"),
            "diagnosis_1": request.form.get("diagnosis_1", ""),
            "diagnosis_2": request.form.get("diagnosis_2", ""),
            "diagnosis_3": request.form.get("diagnosis_3", ""),
            "diagnosis_4": request.form.get("diagnosis_4", ""),
            "diagnosis_5": request.form.get("diagnosis_5", ""),
        }

    try:
        image_bytes = file.read()
        
        # Dual Prediction: 1. RASC-Net Proposed, 2. Soft Voting Ensemble
        rasc_result = predict_rasc_net(image_bytes)
        ensemble_result = predict_ensemble(image_bytes)

        LESION_NAMES = {
            "akiec": "Actinic keratoses",
            "bcc": "Basal cell carcinoma",
            "bkl": "Benign keratosis-like lesions",
            "df": "Dermatofibroma",
            "mel": "Melanoma",
            "nv": "Melanocytic nevi",
            "vasc": "Vascular lesions",
        }

        # Helper to compute top-3 and confidence category
        def format_model_output(res):
            sorted_probs = sorted(res.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)
            top_3 = [
                {
                    "class_code": k,
                    "lesion_name": LESION_NAMES.get(k, k),
                    "probability": float(v),
                    "confidence_pct": round(float(v) * 100, 2),
                }
                for k, v in sorted_probs[:3]
            ]
            conf_val = float(res.get("confidence", 0.0)) * 100.0
            
            if conf_val >= 85.0:
                conf_level = "HIGH"
            elif conf_val >= 60.0:
                conf_level = "MODERATE"
            else:
                conf_level = "LOW"

            return {
                "class_code": res.get("class", ""),
                "lesion_name": LESION_NAMES.get(res.get("class", ""), res.get("class", "")),
                "confidence_pct": round(conf_val, 2),
                "confidence_level": conf_level,
                "probabilities": res.get("probabilities", {}),
                "top_3_predictions": top_3,
                "model_name": res.get("model_used", ""),
            }

        rasc_data = format_model_output(rasc_result)
        ensemble_data = format_model_output(ensemble_result)

        models_agree = (rasc_data["class_code"].lower() == ensemble_data["class_code"].lower())

        # Clinical Risk Engine runs ONLY on RASC-Net Proposed prediction & patient metadata
        risk_assessment = calculate_clinical_risk_score(metadata, rasc_data["class_code"])

        response_payload = {
            "status": "success",
            "models_agree": models_agree,
            "agreement_message": "Both models agree" if models_agree else "Models disagree",
            "rasc_net_proposed": rasc_data,
            "soft_voting_ensemble": ensemble_data,
            "clinical_assessment": risk_assessment,
        }

        return jsonify(response_payload), 200

    except Exception as e:
        print(f"[Predict API Error] {e}")
        return jsonify({"error": str(e)}), 500
