import io
import cv2
import base64
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Blueprint, request, jsonify

from services.inference import get_rasc_net_model, get_label_mapping, preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gradcam_bp = Blueprint('gradcam', __name__)


def find_last_conv_layer(model):
    """Recursively search for the last Conv2D layer in Keras model or nested models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        elif isinstance(layer, tf.keras.models.Model):
            sub_name = find_last_conv_layer(layer)
            if sub_name:
                return sub_name
    return None


def generate_gradcam_overlay(model, input_tensor, original_img_rgb):
    """Compute Grad-CAM heatmap and overlay on original RGB image."""
    last_conv_layer_name = find_last_conv_layer(model)
    if not last_conv_layer_name:
        raise ValueError("No Conv2D layer found in RASC-Net model architecture.")

    logger.info(f"[GradCAM] Using target convolution layer: '{last_conv_layer_name}'")

    # Build multi-output model targeting last conv layer output and model predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        pred_idx = tf.argmax(predictions[0])
        loss = predictions[:, pred_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()
    heatmap_resized = cv2.resize(heatmap_np, (224, 224))

    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Convert original RGB image to BGR for OpenCV blending
    orig_bgr = cv2.cvtColor(original_img_rgb, cv2.COLOR_RGB2BGR)

    # Blend original image and heatmap overlay (60% image, 40% heatmap)
    overlay_bgr = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored_bgr, 0.4, 0)

    # Encode overlay image to Base64 PNG
    _, overlay_buffer = cv2.imencode('.png', overlay_bgr)
    overlay_b64 = "data:image/png;base64," + base64.b64encode(overlay_buffer).decode('utf-8')

    # Encode standalone heatmap image to Base64 PNG
    _, heatmap_buffer = cv2.imencode('.png', heatmap_colored_bgr)
    heatmap_b64 = "data:image/png;base64," + base64.b64encode(heatmap_buffer).decode('utf-8')

    idx2label = get_label_mapping()
    predicted_class = idx2label.get(int(pred_idx), str(int(pred_idx)))
    confidence = float(predictions[0][pred_idx]) * 100.0

    return predicted_class, confidence, heatmap_b64, overlay_b64


@gradcam_bp.route('/gradcam', methods=['POST'])
def gradcam_endpoint():
    """POST /api/gradcam - Generate RASC-Net Grad-CAM visual explainability overlays."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided in request'}), 400

        file = request.files['image']
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({'error': 'Empty image payload received'}), 400

        # Load original image for visualization
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        original_img_rgb = np.array(pil_img, dtype=np.uint8)

        # Preprocess tensor for model input
        _, input_tensor = preprocess_image(image_bytes, "rasc_net")

        # Load thread-safe RASC-Net Keras Singleton
        model = get_rasc_net_model()

        # Compute Grad-CAM
        pred_class, confidence, heatmap_b64, overlay_b64 = generate_gradcam_overlay(
            model, input_tensor, original_img_rgb
        )

        return jsonify({
            'success': True,
            'predicted_class': pred_class,
            'confidence': round(confidence, 2),
            'confidence_pct': round(confidence, 2),
            'gradcam_image': heatmap_b64,
            'overlay_image': overlay_b64,
            'gradcam': overlay_b64,
            'gradcam_image_base64': overlay_b64
        }), 200

    except Exception as e:
        logger.error(f"[GradCAM API Error] Exception during Grad-CAM generation: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Grad-CAM processing error: {str(e)}"
        }), 500
