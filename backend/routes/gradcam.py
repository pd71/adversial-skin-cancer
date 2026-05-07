import io
import base64
from PIL import Image
from flask import Blueprint, request, jsonify
from services.inference import get_models, preprocess_image
from src.run_gradcam import generate_gradcam, find_last_conv_layer, _create_overlay, _preprocess_for

gradcam_bp = Blueprint('gradcam', __name__)

@gradcam_bp.route('/gradcam', methods=['POST'])
def gradcam():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    
    try:
        image_bytes = file.read()
        mobilenet, resnet, idx2label = get_models()
        
        mob_input, img_tensor = preprocess_image(image_bytes, "mobilenetv2")
        mob_last_conv = find_last_conv_layer(mobilenet)
        mob_heatmap, mob_pred, mob_conf = generate_gradcam(
            mobilenet, "MobileNetV2", img_tensor, 
            lambda x: _preprocess_for("mobilenetv2", x),
            mob_last_conv
        )
        
        import numpy as np
        img_array = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))).astype('float32') / 255.0
        
        _, overlay_mob = _create_overlay(img_array, mob_heatmap)
        
        res_input, _ = preprocess_image(image_bytes, "resnet50")
        res_last_conv = find_last_conv_layer(resnet)
        res_heatmap, res_pred, res_conf = generate_gradcam(
            resnet, "ResNet50", img_tensor,
            lambda x: _preprocess_for("resnet50", x),
            res_last_conv
        )
        _, overlay_res = _create_overlay(img_array, res_heatmap)
        
        def encode_img(img_array_uint8):
            img = Image.fromarray(img_array_uint8)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
            
        return jsonify({
            "mobilenet": {
                "overlay": encode_img(overlay_mob),
                "predicted_class": idx2label.get(mob_pred, str(mob_pred)),
                "confidence": float(mob_conf)
            },
            "resnet": {
                "overlay": encode_img(overlay_res),
                "predicted_class": idx2label.get(res_pred, str(res_pred)),
                "confidence": float(res_conf)
            }
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
