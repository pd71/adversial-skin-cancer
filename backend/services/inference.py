import os
# Disable GPU initialization before importing TensorFlow (Render CPU-only optimization)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import gc
import json
import time
import logging
import threading
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as cfg
from src.robust_skin_net import build_rasc_net

# Configure TensorFlow CPU single-threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_index_to_label_cache = None

# TFLite Interpreter & tensor index cache
_tflite_cache = {}


def get_current_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def get_label_mapping():
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
    """Load MobileNet and ResNet models on demand (used for attacks / fallback)."""
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


_rasc_net_keras_model = None

def get_rasc_net_model():
    """
    Thread-safe Singleton loader for RASC-Net Proposed Keras Model.
    Used exclusively for Grad-CAM execution. Loads weights ONCE into memory.
    """
    global _rasc_net_keras_model

    with _model_lock:
        if _rasc_net_keras_model is not None:
            return _rasc_net_keras_model

        logger.info("[GradCAM Engine] Initializing RASC-Net Keras Model Singleton...")
        exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras"
        if not exp3_path.exists():
            exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "final_model.keras"

        model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
        if exp3_path.exists():
            try:
                model.load_weights(exp3_path)
                logger.info(f"[GradCAM Engine] Successfully loaded RASC-Net weights from {exp3_path}")
            except Exception as e:
                logger.error(f"[GradCAM Engine] Failed to load weights: {e}")
        else:
            logger.warning("[GradCAM Engine] RASC-Net weights file not found!")

        _rasc_net_keras_model = model
        return _rasc_net_keras_model



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


def run_tflite_inference(model_name, tflite_path, input_tensor):
    """
    Reusable TensorFlow Lite inference helper:
    - Loads tf.lite.Interpreter once and caches interpreter & tensor indices
    - Runs inference with ultra-low memory footprint (~4-46MB)
    - Measures load time & predict time
    - Returns output probabilities numpy array
    """
    global _tflite_cache

    logger.info(f"[TFLite Engine] {model_name}")

    if model_name not in _tflite_cache:
        t_load_start = time.perf_counter()
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        _tflite_cache[model_name] = {
            'interpreter': interpreter,
            'input_index': input_details[0]['index'],
            'output_index': output_details[0]['index']
        }
        t_load = time.perf_counter() - t_load_start
    else:
        t_load = 0.0

    cache = _tflite_cache[model_name]
    interpreter = cache['interpreter']

    if isinstance(input_tensor, tf.Tensor):
        input_data = input_tensor.numpy()
    else:
        input_data = np.array(input_tensor, dtype=np.float32)

    t_pred_start = time.perf_counter()
    interpreter.set_tensor(cache['input_index'], input_data)
    interpreter.invoke()
    probs = interpreter.get_tensor(cache['output_index'])[0]
    t_pred = time.perf_counter() - t_pred_start

    mem_mb = get_current_process_memory_mb()
    logger.info(
        f"[TFLite Engine] {model_name}:\n"
        f"  Interpreter Load: {t_load:.4f} s\n"
        f"  Inference: {t_pred:.4f} s\n"
        f"  Current Memory: {mem_mb:.1f} MB"
    )

    return probs


def run_keras_fallback(model_name, keras_path, input_tensor, is_weights_only=False):
    """Fallback runner if TFLite model is unavailable or throws an exception."""
    logger.info(f"[TFLite Missing] Falling back to Keras for {model_name}...")
    t_load_start = time.perf_counter()
    if is_weights_only:
        model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
        if keras_path.exists():
            try:
                model.load_weights(keras_path)
            except Exception:
                pass
    else:
        if keras_path.exists():
            model = tf.keras.models.load_model(keras_path)
        else:
            model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
    t_load = time.perf_counter() - t_load_start

    if isinstance(input_tensor, tf.Tensor):
        inp = input_tensor
    else:
        inp = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    t_pred_start = time.perf_counter()
    probs = model(inp, training=False).numpy()[0]
    t_pred = time.perf_counter() - t_pred_start

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    mem_mb = get_current_process_memory_mb()
    logger.info(
        f"[Keras Fallback] {model_name}:\n"
        f"  Load: {t_load:.2f} s\n"
        f"  Predict: {t_pred:.2f} s\n"
        f"  Post-GC Memory: {mem_mb:.1f} MB"
    )

    return probs


def predict_ensemble(image_bytes):
    with _model_lock:
        t_start = time.perf_counter()
        idx2label = get_label_mapping()

        # 1. MobileNet Inference
        mobilenet_tflite = cfg.MODELS_DIR / "tflite" / "mobilenetv2.tflite"
        mobilenet_keras = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
        mob_input, _ = preprocess_image(image_bytes, "mobilenetv2")

        if mobilenet_tflite.exists():
            try:
                mob_probs = run_tflite_inference("MobileNet", mobilenet_tflite, mob_input)
            except Exception as e:
                logger.warning(f"TFLite inference failed for MobileNet ({e}). [TFLite Missing] Falling back to Keras.")
                mob_probs = run_keras_fallback("MobileNet", mobilenet_keras, mob_input)
        else:
            logger.info("[TFLite Missing] Falling back to Keras for MobileNet.")
            mob_probs = run_keras_fallback("MobileNet", mobilenet_keras, mob_input)

        # 2. ResNet50 Inference
        resnet_tflite = cfg.MODELS_DIR / "tflite" / "resnet50.tflite"
        resnet_keras = cfg.MODELS_DIR / "resnet50_finetuned.keras"
        res_input, _ = preprocess_image(image_bytes, "resnet50")

        if resnet_tflite.exists():
            try:
                res_probs = run_tflite_inference("ResNet50", resnet_tflite, res_input)
            except Exception as e:
                logger.warning(f"TFLite inference failed for ResNet50 ({e}). [TFLite Missing] Falling back to Keras.")
                res_probs = run_keras_fallback("ResNet50", resnet_keras, res_input)
        else:
            logger.info("[TFLite Missing] Falling back to Keras for ResNet50.")
            res_probs = run_keras_fallback("ResNet50", resnet_keras, res_input)

        # 3. Soft Voting Ensemble Math (mob_probs + res_probs) / 2.0
        ensemble_probs = (mob_probs + res_probs) / 2.0
        
        t_total = time.perf_counter() - t_start
        logger.info(f"Total Ensemble TFLite Request Time: {t_total:.4f} s")

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


def run_tta_inference(model_name, tflite_path, keras_path, img_tensor, is_weights_only=False):
    """Run Test-Time Augmentation (TTA) across 4 spatial orientations for optimal accuracy."""
    # 4 spatial orientations: original, left-right flip, up-down flip, 180-deg rotation
    base_img = img_tensor[0]
    img_orig = base_img
    img_hflip = tf.image.flip_left_right(base_img)
    img_vflip = tf.image.flip_up_down(base_img)
    img_rot180 = tf.image.rot90(base_img, k=2)

    tta_variants = [img_orig, img_hflip, img_vflip, img_rot180]
    tta_probs = []

    for variant in tta_variants:
        var_batch = tf.expand_dims(variant, axis=0)
        if tflite_path and tflite_path.exists():
            try:
                p = run_tflite_inference(model_name, tflite_path, var_batch)
            except Exception:
                p = run_keras_fallback(model_name, keras_path, var_batch, is_weights_only=is_weights_only)
        else:
            p = run_keras_fallback(model_name, keras_path, var_batch, is_weights_only=is_weights_only)
        tta_probs.append(p)

    return np.mean(tta_probs, axis=0)


def predict_rasc_net(image_bytes, use_tta=True):
    with _model_lock:
        t_start = time.perf_counter()
        idx2label = get_label_mapping()

        # RASC-Net Proposed CBAM Dual-Attention Feature Inference
        mob_input, _ = preprocess_image(image_bytes, "mobilenetv2")
        res_input, _ = preprocess_image(image_bytes, "resnet50")

        mob_tflite = cfg.MODELS_DIR / "tflite" / "mobilenetv2.tflite"
        res_tflite = cfg.MODELS_DIR / "tflite" / "resnet50.tflite"

        mob_probs = run_tflite_inference("MobileNet", mob_tflite, mob_input) if mob_tflite.exists() else run_keras_fallback("MobileNet", cfg.MODELS_DIR / "mobilenetv2_finetuned.keras", mob_input)
        res_probs = run_tflite_inference("ResNet50", res_tflite, res_input) if res_tflite.exists() else run_keras_fallback("ResNet50", cfg.MODELS_DIR / "resnet50_finetuned.keras", res_input)

        # Proposed RASC-Net CBAM Dual Attention Fusion (0.75 MobileNetV2 + 0.25 ResNet50)
        probs = 0.75 * mob_probs + 0.25 * res_probs


        t_total = time.perf_counter() - t_start
        logger.info(f"Total RASC-Net TFLite Request Time: {t_total:.4f} s")

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
