import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as cfg

_mobilenet_model = None
_resnet_model = None
_index_to_label = None

def get_models():
    global _mobilenet_model, _resnet_model, _index_to_label
    
    if _mobilenet_model is not None:
        return _mobilenet_model, _resnet_model, _index_to_label

    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    mobilenet_mapping_path = cfg.MODELS_DIR / "mobilenetv2_class_mapping.json"
    
    _mobilenet_model = tf.keras.models.load_model(mobilenet_path)
    _resnet_model = tf.keras.models.load_model(resnet_path)

    if mobilenet_mapping_path.exists():
        with open(mobilenet_mapping_path, 'r') as f:
            mapping = json.load(f)
            _index_to_label = {int(k): v for k, v in mapping['index_to_label'].items()}
    else:
        _index_to_label = {i: c for i, c in enumerate(cfg.CLASS_NAMES)}
        
    return _mobilenet_model, _resnet_model, _index_to_label

def preprocess_image(image_bytes, model_name):
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

def predict_ensemble(image_bytes):
    mobilenet, resnet, idx2label = get_models()
    
    mob_input, _ = preprocess_image(image_bytes, "mobilenetv2")
    res_input, _ = preprocess_image(image_bytes, "resnet50")
    
    mob_probs = mobilenet(mob_input, training=False).numpy()[0]
    res_probs = resnet(res_input, training=False).numpy()[0]
    
    ensemble_probs = (mob_probs + res_probs) / 2.0
    
    pred_idx = int(np.argmax(ensemble_probs))
    pred_label = idx2label.get(pred_idx, str(pred_idx))
    confidence = float(ensemble_probs[pred_idx])
    
    probs_dict = {idx2label.get(i, str(i)): float(p) for i, p in enumerate(ensemble_probs)}
    
    return {
        "class": pred_label,
        "confidence": confidence,
        "probabilities": probs_dict,
        "model_used": "MobileNetV2 + ResNet50 Ensemble",
        "ensemble_prediction": True
    }


def predict_rasc_net(image_bytes):
    """Run prediction using RASC-Net custom model."""
    rasc_path = cfg.MODELS_DIR / "rasc_net_finetuned.keras"
    if not rasc_path.exists():
        rasc_path = cfg.MODELS_DIR / "rasc_net_best.keras"
    
    if rasc_path.exists():
        rasc_model = tf.keras.models.load_model(rasc_path)
    else:
        # Fallback to ensemble prediction if model checkpoint not trained yet
        return predict_ensemble(image_bytes)

    _, img_tensor = preprocess_image(image_bytes, "rasc_net")
    probs = rasc_model(img_tensor, training=False).numpy()[0]
    
    _, _, idx2label = get_models()
    pred_idx = int(np.argmax(probs))
    pred_label = idx2label.get(pred_idx, str(pred_idx))
    confidence = float(probs[pred_idx])
    probs_dict = {idx2label.get(i, str(i)): float(p) for i, p in enumerate(probs)}

    return {
        "class": pred_label,
        "confidence": confidence,
        "probabilities": probs_dict,
        "model_used": "RASC-Net (Custom Architecture)",
        "ensemble_prediction": False
    }

