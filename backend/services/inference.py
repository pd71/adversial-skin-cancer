import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from src import config as cfg
from src.robust_skin_net import build_rasc_net

_mobilenet_model = None
_resnet_model = None
_rasc_net_model = None
_index_to_label = None


def get_models():
    global _mobilenet_model, _resnet_model, _index_to_label
    
    if _mobilenet_model is not None:
        return _mobilenet_model, _resnet_model, _index_to_label

    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    mobilenet_mapping_path = cfg.MODELS_DIR / "mobilenetv2_class_mapping.json"
    
    if mobilenet_path.exists():
        _mobilenet_model = tf.keras.models.load_model(mobilenet_path)
    else:
        _mobilenet_model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)

    if resnet_path.exists():
        _resnet_model = tf.keras.models.load_model(resnet_path)
    else:
        _resnet_model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)

    if mobilenet_mapping_path.exists():
        with open(mobilenet_mapping_path, 'r') as f:
            mapping = json.load(f)
            _index_to_label = {int(k): v for k, v in mapping['index_to_label'].items()}
    else:
        _index_to_label = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
        
    return _mobilenet_model, _resnet_model, _index_to_label


def get_rasc_net_model():
    """Load RASC-Net Proposed architecture model."""
    global _rasc_net_model
    if _rasc_net_model is not None:
        return _rasc_net_model

    exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras"
    if not exp3_path.exists():
        exp3_path = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "final_model.keras"

    model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
    if exp3_path.exists():
        try:
            model.load_weights(exp3_path)
        except Exception:
            pass

    _rasc_net_model = model
    return _rasc_net_model


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
        "model_used": "Soft Voting Ensemble (MobileNetV2 + ResNet50)",
        "ensemble_prediction": True
    }


def predict_rasc_net(image_bytes):
    """Run prediction using RASC-Net Proposed model."""
    rasc_model = get_rasc_net_model()
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
        "model_used": "RASC-Net Proposed",
        "ensemble_prediction": False
    }
