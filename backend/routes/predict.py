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

    # Optional Patient Metadata
    metadata_raw = request.form.get('metadata')
    metadata = {}
    if metadata_raw:
        try:
            metadata = json.loads(metadata_raw)
        except Exception:
            metadata = {}
    else:
        # Extract direct form fields if available
        metadata = {
            "age": request.form.get("age", 0),
            "sex": request.form.get("sex", "Other"),
            "location": request.form.get("location", "Other"),
            "duration": request.form.get("duration", "Unknown"),
            "symptoms": request.form.get("symptoms", "None"),
            "family_history": request.form.get("family_history", "No"),
            "previous_cancer": request.form.get("previous_cancer", "No"),
            "notes": request.form.get("notes", ""),
        }

    try:
        image_bytes = file.read()
        
        # CNN Inference runs ONLY on image_bytes
        result = predict_rasc_net(image_bytes)

        # Rule-Based Clinical Risk Score (Decision Support System)
        risk_assessment = calculate_clinical_risk_score(metadata, result.get("class", ""))

        # Extract Top 3 predictions
        sorted_probs = sorted(result.get("probabilities", {}).items(), key=lambda x: x[1], reverse=True)
        top_3 = [{"class_code": k, "probability": float(v)} for k, v in sorted_probs[:3]]

        # Determine Confidence Level Category
        conf = float(result.get("confidence", 0.0))
        if conf >= 0.90:
            conf_level = "HIGH"
            conf_msg = "Model exhibits high classification certainty."
        elif conf >= 0.70:
            conf_level = "MODERATE"
            conf_msg = "Model exhibits moderate classification certainty."
        else:
            conf_level = "LOW"
            conf_msg = "Manual dermatological examination is strongly recommended due to low AI confidence."

        result["top_3_predictions"] = top_3
        result["confidence_level"] = conf_level
        result["confidence_message"] = conf_msg

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

