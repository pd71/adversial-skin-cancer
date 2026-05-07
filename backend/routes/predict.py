from flask import Blueprint, request, jsonify
from services.inference import predict_ensemble

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image_bytes = file.read()
        result = predict_ensemble(image_bytes)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
