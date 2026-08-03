import os
# Disable GPU initialization before importing TensorFlow (Render CPU-only optimization)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import gc
import json
import time
import logging
import threading
import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as cfg
from src.robust_skin_net import build_rasc_net

# Configure TensorFlow to use minimal CPU resources (single-threaded execution)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_index_to_label_cache = None


def get_current_process_memory_mb():
    """Helper to return current process RSS memory in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_label_mapping():
    """Load and cache label index to string mapping without loading ML models into RAM."""
    global _index_to_label_cache
    if _index_to_label_cache is not None:
        return _index_to_label_cache

    mobilenet_mapping_path = cfg.MODELS_DIR / "mobilenetv2_class_mapping.json"
    if mobilenet_mapping_path.exists():
        try:
            with open(mobilenet_mapping_path, 'r') as f:
                mapping = json.load(f)
                _index_to_label_cache = {int(k): v for k, v in mapping['index_to_label'].items()}
        except Exception:
            _index_to_label_cache = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
    else:
        _index_to_label_cache = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}

    return _index_to_label_cache


def get_models():
    """Load MobileNet and ResNet models on demand. Reused if available."""
    idx2label = get_label_mapping()
    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"

    if mobilenet_path.exists():
        mobilenet = tf.keras.models.load_model(mobilenet_path)
    else:
        mobilenet = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)

    if resnet_path.exists():
        resnet = tf.keras.models.load_model(resnet_path)
    else:
        resnet = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)

    return mobilenet, resnet, idx2label


def get_rasc_net_model():
    """Load RASC-Net Proposed architecture model on demand."""
    exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras"
    if not exp3_path.exists():
        exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "final_model.keras"

    model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
    if exp3_path.exists():
        try:
            model.load_weights(exp3_path)
        except Exception as e:
            logger.warning(f"Failed to load RASC-Net weights: {e}")

    return model


def preprocess_image(image_bytes, model_name="rasc_net"):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(cfg.IMAGE_SIZE)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_255 = img_tensor * 255.0
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_255), img_tensor
    elif model_name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input(img_255), img_tensor
    return img_tensor, img_tensor


def run_model_inference(model_key_name, model_path, img_input, is_weights_only=False):
    """
    Reusable helper that:
    1. Measures model load time
    2. Measures inference execution time
    3. Cleans up RAM via del, clear_session, and gc.collect()
    4. Logs current RSS memory footprint
    5. Returns probabilities array, load_time, predict_time
    """
    t_load_start = time.perf_counter()
    if is_weights_only:
        model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
        if model_path.exists():
            try:
                model.load_weights(model_path)
            except Exception as e:
                logger.warning(f"Failed to load weights for {model_key_name}: {e}")
    else:
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
        else:
            model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
    t_load_end = time.perf_counter()
    load_time = t_load_end - t_load_start

    t_pred_start = time.perf_counter()
    probs = model(img_input, training=False).numpy()[0]
    t_pred_end = time.perf_counter()
    predict_time = t_pred_end - t_pred_start

    # Immediate Memory & Session Cleanup
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    mem_mb = get_current_process_memory_mb()
    logger.info(
        f"\n{model_key_name}:\n"
        f"  Load: {load_time:.2f} s\n"
        f"  Predict: {predict_time:.2f} s\n"
        f"  Post-GC Memory: {mem_mb:.1f} MB"
    )

    return probs, load_time, predict_time


def predict_ensemble(image_bytes):
    with _model_lock:
        t_req_start = time.perf_counter()
        idx2label = get_label_mapping()

        mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
        mob_input, _ = preprocess_image(image_bytes, "mobilenetv2")
        mob_probs, mob_load, mob_pred = run_model_inference("MobileNet", mobilenet_path, mob_input)

        resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
        res_input, _ = preprocess_image(image_bytes, "resnet50")
        res_probs, res_load, res_pred = run_model_inference("ResNet", resnet_path, res_input)

        # Soft Voting Ensemble (Identical Math: (mob_probs + res_probs) / 2.0)
        ensemble_probs = (mob_probs + res_probs) / 2.0
        
        t_req_end = time.perf_counter()
        total_time = t_req_end - t_req_start

        logger.info(f"\nTotal Ensemble Request: {total_time:.2f} s")

        pred_idx = int(np.argmax(ensemble_probs))
        pred_label = idx2label.get(pred_idx, str(pred_idx))
        confidence = float(ensemble_probs[pred_idx])
        probs_dict = {idx2label.get(i, str(i)): float(p) for i, p in enumerate(ensemble_probs)}
        
        return {
            "class": pred_label,
            "confidence": confidence,
            "probabilities": probs_dict,
            "model_used": "Soft Voting Ensemble (MobileNetV2 + ResNet50)",
            "ensemble_prediction": True
        }


def predict_rasc_net(image_bytes):
    with _model_lock:
        t_req_start = time.perf_counter()
        idx2label = get_label_mapping()

        exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras"
        if not exp3_path.exists():
            exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "final_model.keras"

        _, img_tensor = preprocess_image(image_bytes, "rasc_net")
        probs, rasc_load, rasc_pred = run_model_inference("RASC-Net", exp3_path, img_tensor, is_weights_only=True)

        t_req_end = time.perf_counter()
        total_time = t_req_end - t_req_start

        logger.info(f"\nTotal RASC-Net Request: {total_time:.2f} s")

        pred_idx = int(np.argmax(probs))
        pred_label = idx2label.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])
        probs_dict = {idx2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}

        return {
            "class": pred_label,
            "confidence": confidence,
            "probabilities": probs_dict,
            "model_used": "RASC-Net Proposed",
            "ensemble_prediction": False
        }
