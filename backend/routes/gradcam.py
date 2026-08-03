import io
import gc
import os
import sys
import time
import base64
import logging
import threading
import traceback
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from flask import Blueprint, request, jsonify

from services.inference import get_rasc_net_model, get_label_mapping, get_current_process_memory_mb
from src.run_gradcam import find_last_conv_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gradcam_bp = Blueprint('gradcam', __name__)


def log_step(step_num, step_name, status, t_start, m_start):
    pid = os.getpid()
    tid = threading.get_ident()
    t_now = time.perf_counter()
    m_now = get_current_process_memory_mb()
    dt = t_now - t_start
    dm = m_now - m_start
    logger.info(
        f"[GRADCAM STEP {step_num} - PID:{pid} TID:{tid}] {step_name} -> Status: {status} | "
        f"RSS: {m_now:.2f} MB (Delta: {dm:+.2f} MB) | Step Time: {dt:.4f}s"
    )
    return t_now, m_now


@gradcam_bp.route('/gradcam', methods=['POST'])
def gradcam():
    t0 = time.perf_counter()
    m0 = get_current_process_memory_mb()
    pid = os.getpid()
    tid = threading.get_ident()

    logger.info(f"[GRADCAM START - PID:{pid} TID:{tid}] Entering /api/gradcam route | Initial RSS: {m0:.2f} MB")

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided in request", "status": "error"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected image file", "status": "error"}), 400
    
    rasc_model = None
    grad_model = None

    try:
        # Step 0: Read Image
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image file received", "status": "error"}), 400
        log_step(0, "read_image_bytes", "SUCCESS", t_sub, m_sub)

        # Step 1: get_rasc_net_model()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        rasc_model = get_rasc_net_model()
        log_step(1, "get_rasc_net_model()", "SUCCESS", t_sub, m_sub)

        # Step 2: get_label_mapping()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        idx2label = get_label_mapping()
        log_step(2, "get_label_mapping()", "SUCCESS", t_sub, m_sub)

        # Step 3: preprocess_image()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        raw_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        img_array = np.array(raw_pil).astype('float32') / 255.0
        input_tensor = tf.expand_dims(tf.convert_to_tensor(img_array, dtype=tf.float32), axis=0)
        log_step(3, "preprocess_image()", "SUCCESS", t_sub, m_sub)

        # Step 4: find_last_conv_layer()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        try:
            last_conv_layer_name = find_last_conv_layer(rasc_model)
        except Exception as e:
            logger.warning(f"[GRADCAM] find_last_conv_layer warning: {e}. Fallback to conv2d_20.")
            last_conv_layer_name = "conv2d_20"
        log_step(4, f"find_last_conv_layer() -> '{last_conv_layer_name}'", "SUCCESS", t_sub, m_sub)

        # Step 5: model.get_layer()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        target_layer = rasc_model.get_layer(last_conv_layer_name)
        log_step(5, f"model.get_layer('{last_conv_layer_name}')", "SUCCESS", t_sub, m_sub)

        # Step 6: tf.keras.models.Model(...) [Sub-Model Construction]
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        logger.info(f"[GRADCAM HIGH-RISK] BEFORE tf.keras.models.Model(...) | RSS: {m_sub:.2f} MB")
        grad_model = tf.keras.models.Model(
            inputs=rasc_model.inputs,
            outputs=[target_layer.output, rasc_model.output]
        )
        log_step(6, "tf.keras.models.Model(...) [grad_model construction]", "SUCCESS", t_sub, m_sub)

        # Step 7: Entering tf.GradientTape()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        logger.info(f"[GRADCAM HIGH-RISK] BEFORE Entering tf.GradientTape() | RSS: {m_sub:.2f} MB")

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            
            # Step 8: grad_model(input_tensor) forward pass
            t_fwd, m_fwd = time.perf_counter(), get_current_process_memory_mb()
            logger.info(f"[GRADCAM HIGH-RISK] BEFORE grad_model(input_tensor) | RSS: {m_fwd:.2f} MB")
            conv_outputs, predictions = grad_model(input_tensor, training=False)
            log_step(8, "grad_model(input_tensor) [forward pass]", "SUCCESS", t_fwd, m_fwd)

            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        log_step(7, "GradientTape forward context complete", "SUCCESS", t_sub, m_sub)

        # Step 9: tape.gradient(...) backward pass
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        logger.info(f"[GRADCAM HIGH-RISK] BEFORE tape.gradient(...) | RSS: {m_sub:.2f} MB")
        grads = tape.gradient(class_channel, conv_outputs)
        log_step(9, "tape.gradient(...) [backprop grads]", "SUCCESS", t_sub, m_sub)

        if grads is None:
            logger.error("[GRADCAM ERROR] Computed gradients are None!")
            return jsonify({"error": "Failed to compute gradients for Grad-CAM", "status": "error"}), 500

        # Step 10: Pooled gradients & Heatmap computation
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_0 = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs_0 * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if float(max_val.numpy()) > 0:
            heatmap = heatmap / max_val

        heatmap_np = heatmap.numpy()
        log_step(10, "pooled_grads & heatmap computation", "SUCCESS", t_sub, m_sub)

        # Step 11: cv2.resize()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
        log_step(11, "cv2.resize()", "SUCCESS", t_sub, m_sub)

        # Step 12: cv2.applyColorMap()
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        log_step(12, "cv2.applyColorMap()", "SUCCESS", t_sub, m_sub)

        # Step 13: Image Overlay
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        img_bgr = cv2.cvtColor(np.uint8(255 * img_array), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        log_step(13, "image_overlay [addWeighted]", "SUCCESS", t_sub, m_sub)

        # Step 14: PNG encoding
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        overlay_pil = Image.fromarray(overlay_rgb)
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="PNG")
        log_step(14, "PNG image encoding [BytesIO]", "SUCCESS", t_sub, m_sub)

        # Step 15: Base64 encoding
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        overlay_base64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        log_step(15, "base64.b64encode()", "SUCCESS", t_sub, m_sub)

        pred_idx_int = int(pred_index.numpy())
        predicted_code = idx2label.get(pred_idx_int, str(pred_idx_int))
        confidence = float(predictions[0][pred_index].numpy())

        # Step 16: jsonify() & response
        t_sub, m_sub = time.perf_counter(), get_current_process_memory_mb()
        resp = jsonify({
            "status": "success",
            "gradcam": overlay_base64,
            "gradcam_image_base64": overlay_base64,
            "model_name": "RASC-Net Proposed",
            "last_conv_layer": last_conv_layer_name,
            "predicted_class": predicted_code,
            "confidence_pct": round(confidence * 100.0, 2),
        })
        log_step(16, "jsonify() [response built]", "SUCCESS", t_sub, m_sub)

        m_end = get_current_process_memory_mb()
        t_end = time.perf_counter()
        logger.info(
            f"[GRADCAM COMPLETE - PID:{pid} TID:{tid}] Total Elapsed: {t_end-t0:.4f}s | "
            f"Final RSS: {m_end:.2f} MB (Total Delta: {m_end-m0:+.2f} MB)"
        )
        return resp, 200

    except Exception as e:
        logger.error(
            f"[GRADCAM EXCEPTION TRACE - PID:{pid} TID:{tid}]\n"
            f"  Type: {type(e).__name__}\n"
            f"  Message: {e}\n"
            f"  Traceback:\n{traceback.format_exc()}"
        )
        return jsonify({"error": str(e), "status": "error"}), 500

    finally:
        if grad_model is not None:
            del grad_model
        gc.collect()
