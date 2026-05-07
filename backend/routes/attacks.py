import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Blueprint, request, jsonify

from services.inference import get_models, preprocess_image
from src.run_attacks import _fgsm_attack, _pgd_attack, _cw_simplified_attack, _predict_probs

attacks_bp = Blueprint('attacks', __name__)

def encode_tensor(tensor_img):
    img_array = np.uint8(np.clip(tensor_img.numpy()[0], 0.0, 1.0) * 255.0)
    img = Image.fromarray(img_array)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@attacks_bp.route('/attacks', methods=['POST'])
def attacks():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    
    try:
        image_bytes = file.read()
        mobilenet, _, _ = get_models()
        
        _, img_tensor = preprocess_image(image_bytes, "mobilenetv2")
        
        probs_fn = lambda x: _predict_probs(mobilenet, "mobilenetv2", x)
        clean_probs = probs_fn(img_tensor)
        pred_idx = tf.argmax(clean_probs, axis=1)
        y_true = tf.one_hot(pred_idx, depth=clean_probs.shape[1])
        
        x_fgsm = _fgsm_attack(img_tensor, y_true, probs_fn)
        x_pgd = _pgd_attack(img_tensor, y_true, probs_fn)
        x_cw = _cw_simplified_attack(img_tensor, y_true, probs_fn, steps=2)
        
        return jsonify({
            "original": encode_tensor(img_tensor),
            "fgsm": encode_tensor(x_fgsm),
            "pgd": encode_tensor(x_pgd),
            "cw": encode_tensor(x_cw)
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
