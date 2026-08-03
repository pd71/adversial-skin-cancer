import io
import gc
import base64
import logging
import traceback
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from flask import Blueprint, request, jsonify

from services.inference import get_rasc_net_model, get_label_mapping
from src.run_gradcam import find_last_conv_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gradcam_bp = Blueprint('gradcam', __name__)

@gradcam_bp.route('/gradcam', methods=['POST'])
def gradcam():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request", "status": "error"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image file", "status": "error"}), 400
    
    rasc_model = None
    grad_model = None

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file received", "status": "error"}), 400
        
        # 1. Load RASC-Net Proposed Keras Model (Only loaded during Grad-CAM)
        logger.info("[GradCAM] Loading Keras model...")
        rasc_model = get_rasc_net_model()
        if rasc_model is None:
            return jsonify({"error": "RASC-Net Proposed model checkpoint not found", "status": "error"}), 404

        idx2label = get_label_mapping()

        # 2. Preprocess Image (1, 224, 224, 3) in [0, 1]
        raw_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_array = np.array(raw_pil).astype('float32') / 255.0
        input_tensor = tf.expand_dims(tf.convert_to_tensor(img_array, dtype=tf.float32), axis=0)

        # 3. Dynamically Find & Verify Last Conv2D Layer
        try:
            last_conv_layer_name = find_last_conv_layer(rasc_model)
        except Exception as e:
            logger.warning(f"[GradCAM] Layer auto-detect warning: {e}. Fallback to conv2d_20.")
            last_conv_layer_name = "conv2d_20"

        logger.info(f"[GradCAM] Target Layer: {last_conv_layer_name}")

        # 4. Build Grad-CAM Sub-Model
        target_layer = rasc_model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=rasc_model.inputs,
            outputs=[target_layer.output, rasc_model.output]
        )

        # 5. Compute Gradients via GradientTape
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            return jsonify({"error": "Failed to compute gradients for Grad-CAM", "status": "error"}), 500

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_0 = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs_0 * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if float(max_val.numpy()) > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()

        # 6. Resize Heatmap & Apply Jet ColorMap Overlay
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        img_bgr = cv2.cvtColor(np.uint8(255 * img_array), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # 7. Base64 Encoding
        overlay_pil = Image.fromarray(overlay_rgb)
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="PNG")
        overlay_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

        pred_idx_int = int(pred_index.numpy())
        predicted_code = idx2label.get(pred_idx_int, str(pred_idx_int))
        confidence = float(predictions[0][pred_index].numpy())

        logger.info("[GradCAM] Heatmap generated successfully.")

        return jsonify({
            "status": "success",
            "gradcam": overlay_base64,
            "gradcam_image_base64": overlay_base64,
            "model_name": "RASC-Net Proposed",
            "last_conv_layer": last_conv_layer_name,
            "predicted_class": predicted_code,
            "confidence_pct": round(confidence * 100.0, 2),
        }), 200

    except Exception as e:
        logger.error(f"[GradCAM API Exception] {e}", exc_info=True)
        return jsonify({"error": str(e), "status": "error"}), 500

    finally:
        # 8. Strict Memory Management & Session Cleanup
        if rasc_model is not None:
            del rasc_model
        if grad_model is not None:
            del grad_model
        tf.keras.backend.clear_session()
        gc.collect()
        logger.info("[GradCAM] Cleanup completed.")
