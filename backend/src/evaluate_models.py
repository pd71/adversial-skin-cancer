"""Comprehensive Evaluation Script for Skin Cancer Models & Defense Pipeline.

Calculates dynamic evaluation metrics on HAM10000 dataset:
- Dataset distribution & class weights
- Clean test evaluation (Accuracy, Macro F1, Weighted Precision, Weighted Recall, Loss, Confusion Matrix, Per-class metrics, ROC-AUC, PR curves)
- Adversarial evaluation (FGSM, PGD, CW attacks: Robust Acc, Acc Drop, Attack Success Rate, Misclassification %, Confidence Drop)
- Defense pipeline evaluation (Feature Squeezing + Gaussian Blur + JPEG Compression against FGSM, PGD, CW: Before, After, Abs Gain, Rel Gain, Recovery Rate, Remaining Gap)
- Exports complete results to JSON and CSVs.
"""

import gc
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_class_weight

import config as cfg
from data_loader import load_ham10000_data


# Attack Hyperparameters
EVAL_SAMPLES = 50
FGSM_EPS = 0.03
PGD_EPS = 0.03
PGD_ALPHA = 0.007
PGD_STEPS = 5
CW_C = 0.1
CW_STEPS = 10
CW_LR = 0.01

LESION_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}


def clear_memory() -> None:
    tf.keras.backend.clear_session()
    gc.collect()


def get_dataset_stats() -> Dict[str, Any]:
    """Calculate HAM10000 dataset sample counts, percentages, and balanced weights."""
    if not cfg.METADATA_PATH.exists():
        return {}
    
    df = pd.read_csv(cfg.METADATA_PATH)
    total_samples = len(df)
    counts = df[cfg.TARGET_COLUMN].value_counts().to_dict()
    
    classes = sorted(list(LESION_NAMES.keys()))
    y_all = df[cfg.TARGET_COLUMN].values
    weights = compute_class_weight(class_weight="balanced", classes=np.array(classes), y=y_all)
    weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
    
    class_stats = []
    for cls in classes:
        sample_cnt = int(counts.get(cls, 0))
        pct = float((sample_cnt / total_samples) * 100.0) if total_samples > 0 else 0.0
        class_stats.append({
            "class_code": cls,
            "lesion_name": LESION_NAMES.get(cls, cls),
            "sample_count": sample_cnt,
            "percentage": round(pct, 2),
            "class_weight": round(weight_dict.get(cls, 1.0), 3),
        })
    
    # Sort by sample count descending for layout display matching reference image
    class_stats.sort(key=lambda x: x["sample_count"], reverse=True)
    
    return {
        "total_samples": total_samples,
        "num_classes": len(classes),
        "class_distribution": class_stats,
    }


def preprocess_for_model(model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    images_255 = images_01 * 255.0
    if model_name.lower() in ["mobilenetv2", "mobilenet"]:
        return tf.keras.applications.mobilenet_v2.preprocess_input(images_255)
    if model_name.lower() in ["resnet50", "resnet"]:
        return tf.keras.applications.resnet.preprocess_input(images_255)
    return images_01


def predict_model_probs(model: tf.keras.Model, model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    inputs = preprocess_for_model(model_name, images_01)
    return model(inputs, training=False)


def ensemble_predict_probs(mobilenet: tf.keras.Model, resnet: tf.keras.Model, images_01: tf.Tensor) -> tf.Tensor:
    m_probs = predict_model_probs(mobilenet, "mobilenetv2", images_01)
    r_probs = predict_model_probs(resnet, "resnet50", images_01)
    return (m_probs + r_probs) / 2.0


# --- Adversarial Attack Functions ---

def fgsm_attack(x: tf.Tensor, y: tf.Tensor, probs_fn: Any, eps: float = FGSM_EPS) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = probs_fn(x)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, probs))
    grad = tape.gradient(loss, x)
    x_adv = x + eps * tf.sign(grad)
    return tf.clip_by_value(x_adv, 0.0, 1.0)


def pgd_attack(x: tf.Tensor, y: tf.Tensor, probs_fn: Any, eps: float = PGD_EPS, alpha: float = PGD_ALPHA, steps: int = PGD_STEPS) -> tf.Tensor:
    x_orig = tf.identity(x)
    x_adv = tf.identity(x)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            probs = probs_fn(x_adv)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, probs))
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x_orig - eps, x_orig + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
    return x_adv


def cw_attack(x: tf.Tensor, y: tf.Tensor, probs_fn: Any, c: float = CW_C, steps: int = CW_STEPS, lr: float = CW_LR) -> tf.Tensor:
    x_orig = tf.identity(x)
    x_adv = tf.Variable(x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            probs = probs_fn(x_adv)
            true_prob = tf.reduce_sum(y * probs, axis=1)
            other_prob = tf.reduce_max((1.0 - y) * probs, axis=1)
            f_loss = tf.nn.relu(true_prob - other_prob)
            l2_loss = tf.reduce_sum(tf.square(x_adv - x_orig), axis=[1, 2, 3])
            loss = tf.reduce_mean(l2_loss + c * f_loss)
        grads = tape.gradient(loss, x_adv)
        optimizer.apply_gradients([(grads, x_adv)])
        x_adv.assign(tf.clip_by_value(x_adv, 0.0, 1.0))
    return tf.convert_to_tensor(x_adv)


# --- Defense Pipeline ---

def bit_depth_reduction(images: tf.Tensor, bits: int = 4) -> tf.Tensor:
    levels = float(2**bits - 1)
    squeezed = tf.round(images * levels) / levels
    return tf.clip_by_value(squeezed, 0.0, 1.0)


def gaussian_blur(images: tf.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> tf.Tensor:
    coords = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1) / 2.0
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    kernel_2d = tf.tensordot(g, g, axes=0)[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel_2d, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.clip_by_value(blurred, 0.0, 1.0)


def jpeg_compression_defense(images: tf.Tensor, quality: int = 70) -> tf.Tensor:
    compressed = []
    for i in range(images.shape[0]):
        img = tf.cast(images[i] * 255.0, tf.uint8)
        jpeg_bytes = tf.io.encode_jpeg(img, quality=quality)
        decoded = tf.io.decode_jpeg(jpeg_bytes, channels=3)
        compressed.append(tf.cast(decoded, tf.float32) / 255.0)
    return tf.stack(compressed, axis=0)


def apply_defense_pipeline(images: tf.Tensor) -> tf.Tensor:
    x = bit_depth_reduction(images, bits=4)
    x = gaussian_blur(x, kernel_size=3, sigma=1.0)
    x = jpeg_compression_defense(x, quality=70)
    return tf.clip_by_value(x, 0.0, 1.0)


# --- Evaluation Logic ---

def evaluate_clean_performance(
    model_name: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    
    try:
        ce_loss = float(log_loss(y_true, probs))
    except Exception:
        ce_loss = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()

    # Per-class metrics
    p_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    class_metrics = []
    roc_curves = {}
    pr_curves = {}

    for idx, cls in enumerate(class_names):
        cls_mask = (y_true == idx)
        cls_acc = float(np.mean(y_pred[cls_mask] == idx)) if np.sum(cls_mask) > 0 else 0.0
        support = int(np.sum(cls_mask))
        
        class_metrics.append({
            "class_code": cls,
            "lesion_name": LESION_NAMES.get(cls, cls),
            "accuracy": round(cls_acc * 100.0, 2),
            "precision": round(float(p_per_class[idx]) * 100.0, 2),
            "recall": round(float(r_per_class[idx]) * 100.0, 2),
            "f1_score": round(float(f1_per_class[idx]) * 100.0, 2),
            "support": support,
        })

        # ROC Curve per class
        y_binary = (y_true == idx).astype(int)
        cls_probs = probs[:, idx]
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, cls_probs)
            roc_auc_score = float(auc(fpr, tpr))
            # Sample points for light JSON payloads
            step = max(1, len(fpr) // 20)
            roc_curves[cls] = {
                "fpr": [round(float(v), 4) for v in fpr[::step]],
                "tpr": [round(float(v), 4) for v in tpr[::step]],
                "auc": round(roc_auc_score, 4),
            }

            # PR Curve per class
            prec, rec, _ = precision_recall_curve(y_binary, cls_probs)
            ap_score = float(average_precision_score(y_binary, cls_probs))
            pr_step = max(1, len(prec) // 20)
            pr_curves[cls] = {
                "precision": [round(float(v), 4) for v in prec[::pr_step]],
                "recall": [round(float(v), 4) for v in rec[::pr_step]],
                "ap": round(ap_score, 4),
            }

    return {
        "model_name": model_name,
        "clean_accuracy": round(acc * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "weighted_precision": round(weighted_p * 100.0, 2),
        "weighted_recall": round(weighted_r * 100.0, 2),
        "cross_entropy_loss": round(ce_loss, 4),
        "confusion_matrix": cm,
        "class_metrics": class_metrics,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }


def run_full_evaluation() -> Dict[str, Any]:
    print("=== Launching Full Model & Defense Evaluation Pipeline ===")
    cfg.ensure_project_dirs()
    
    # 1. Dataset stats
    dataset_stats = get_dataset_stats()
    print(f"Loaded dataset stats: {dataset_stats['total_samples']} samples across {dataset_stats['num_classes']} classes.")

    # 2. Load HAM10000 Data
    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE)
    test_ds = data_dict["test_ds"]
    test_df = data_dict["test_df"]
    index_to_label = data_dict["index_to_label"]
    class_names = [index_to_label[i] for i in sorted(index_to_label.keys())]

    # Load model checkpoints
    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"

    if not mobilenet_path.exists() or not resnet_path.exists():
        raise FileNotFoundError("Required model files missing in backend/models.")

    mobilenet = tf.keras.models.load_model(mobilenet_path)
    resnet = tf.keras.models.load_model(resnet_path)

    # Clean full test set predictions
    print("\n[CLEAN EVAL] Generating full test set predictions...")
    m_probs_clean = mobilenet.predict(test_ds, verbose=0)
    r_probs_clean = resnet.predict(test_ds, verbose=0)
    ens_probs_clean = (m_probs_clean + r_probs_clean) / 2.0
    y_test_full = test_df["label"].values.astype(int)

    clean_evals = {
        "MobileNetV2": evaluate_clean_performance("MobileNetV2", y_test_full, m_probs_clean, class_names),
        "ResNet50": evaluate_clean_performance("ResNet50", y_test_full, r_probs_clean, class_names),
        "Soft Voting Ensemble": evaluate_clean_performance("Soft Voting Ensemble", y_test_full, ens_probs_clean, class_names),
    }

    # Extract test subset for adversarial & defense testing
    print(f"\n[ADV & DEFENSE EVAL] Extracting subset of {EVAL_SAMPLES} test samples...")
    subset_ds = test_ds.unbatch().take(EVAL_SAMPLES).batch(EVAL_SAMPLES)
    x_sub, y_sub_oh = next(iter(subset_ds))
    x_sub = tf.cast(x_sub, tf.float32)
    y_sub = tf.argmax(y_sub_oh, axis=1).numpy()

    model_probs_fns = {
        "MobileNetV2": lambda img: predict_model_probs(mobilenet, "mobilenetv2", img),
        "ResNet50": lambda img: predict_model_probs(resnet, "resnet50", img),
        "Soft Voting Ensemble": lambda img: ensemble_predict_probs(mobilenet, resnet, img),
    }

    attack_fns = {
        "FGSM": lambda img, y_oh, fn: fgsm_attack(img, y_oh, fn),
        "PGD": lambda img, y_oh, fn: pgd_attack(img, y_oh, fn),
        "CW": lambda img, y_oh, fn: cw_attack(img, y_oh, fn),
    }

    adv_results = []
    defense_results = []

    for m_name, probs_fn in model_probs_fns.items():
        clean_probs_sub = probs_fn(x_sub).numpy()
        clean_preds_sub = np.argmax(clean_probs_sub, axis=1)
        clean_acc_sub = float(np.mean(clean_preds_sub == y_sub))
        
        # True class confidence on clean images
        clean_true_confs = clean_probs_sub[np.arange(len(y_sub)), y_sub]

        for a_name, attack_fn in attack_fns.items():
            print(f"Evaluating {m_name} under {a_name} attack...")
            x_adv = attack_fn(x_sub, y_sub_oh, probs_fn)
            adv_probs = probs_fn(x_adv).numpy()
            adv_preds = np.argmax(adv_probs, axis=1)
            
            robust_acc = float(np.mean(adv_preds == y_sub))
            acc_drop = clean_acc_sub - robust_acc
            misclass_pct = (1.0 - robust_acc) * 100.0
            
            # Attack success rate: among clean-correct, proportion misclassified after attack
            correct_mask = (clean_preds_sub == y_sub)
            if np.sum(correct_mask) > 0:
                asr = float(np.mean(adv_preds[correct_mask] != y_sub[correct_mask])) * 100.0
            else:
                asr = 100.0
                
            # Confidence drop on true class
            adv_true_confs = adv_probs[np.arange(len(y_sub)), y_sub]
            conf_reduction = float(np.mean(np.maximum(0.0, clean_true_confs - adv_true_confs))) * 100.0

            adv_results.append({
                "model_name": m_name,
                "attack_type": a_name,
                "clean_accuracy": round(clean_acc_sub * 100.0, 2),
                "robust_accuracy": round(robust_acc * 100.0, 2),
                "accuracy_drop": round(acc_drop * 100.0, 2),
                "attack_success_rate": round(asr, 2),
                "misclassification_pct": round(misclass_pct, 2),
                "confidence_reduction": round(conf_reduction, 2),
            })

            # Defense Pipeline Application
            print(f"Applying defense pipeline for {m_name} under {a_name}...")
            x_def = apply_defense_pipeline(x_adv)
            def_probs = probs_fn(x_def).numpy()
            def_preds = np.argmax(def_probs, axis=1)
            
            acc_before = robust_acc
            acc_after = float(np.mean(def_preds == y_sub))
            abs_gain = acc_after - acc_before
            rel_gain = (abs_gain / (acc_before + 1e-8)) * 100.0
            
            clean_ref = clean_acc_sub
            denom = clean_ref - acc_before
            norm_recovery = (abs_gain / denom * 100.0) if denom > 1e-5 else 0.0
            remaining_gap = clean_ref - acc_after

            defense_results.append({
                "model_name": m_name,
                "attack_type": a_name,
                "accuracy_before_defense": round(acc_before * 100.0, 2),
                "accuracy_after_defense": round(acc_after * 100.0, 2),
                "absolute_gain": round(abs_gain * 100.0, 2),
                "relative_gain": round(rel_gain, 2),
                "normalized_recovery_rate": round(max(0.0, norm_recovery), 2),
                "remaining_accuracy_gap": round(max(0.0, remaining_gap * 100.0), 2),
            })

    clear_memory()

    full_payload = {
        "dataset_stats": dataset_stats,
        "clean_evaluations": clean_evals,
        "adversarial_evaluations": adv_results,
        "defense_evaluations": defense_results,
    }

    # Save output JSON
    json_path = cfg.METRICS_DIR / "full_evaluation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    print(f"\n[SUCCESS] Exported full evaluation JSON to: {json_path}")

    # Export CSVs for backward compatibility
    adv_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Clean_Accuracy": r["clean_accuracy"] / 100.0,
            "FGSM_Accuracy": r["robust_accuracy"] / 100.0 if r["attack_type"] == "FGSM" else None,
            "PGD_Accuracy": r["robust_accuracy"] / 100.0 if r["attack_type"] == "PGD" else None,
            "CW_Accuracy": r["robust_accuracy"] / 100.0 if r["attack_type"] == "CW" else None,
        }
        for r in adv_results
    ])
    
    # Pivot adv_df by Model
    pivoted_adv = []
    for m in ["MobileNetV2", "ResNet50", "Soft Voting Ensemble"]:
        m_rows = [r for r in adv_results if r["model_name"] == m]
        clean_val = m_rows[0]["clean_accuracy"] / 100.0 if m_rows else 0.0
        fgsm_val = next((r["robust_accuracy"] / 100.0 for r in m_rows if r["attack_type"] == "FGSM"), 0.0)
        pgd_val = next((r["robust_accuracy"] / 100.0 for r in m_rows if r["attack_type"] == "PGD"), 0.0)
        cw_val = next((r["robust_accuracy"] / 100.0 for r in m_rows if r["attack_type"] == "CW"), 0.0)
        pivoted_adv.append({
            "Model": m,
            "Clean_Accuracy": clean_val,
            "FGSM_Accuracy": fgsm_val,
            "PGD_Accuracy": pgd_val,
            "CW_Accuracy": cw_val,
        })
    pd.DataFrame(pivoted_adv).to_csv(cfg.METRICS_DIR / "adversarial_results.csv", index=False)

    def_df = pd.DataFrame([
        {
            "Model": r["model_name"],
            "Attack": r["attack_type"],
            "Accuracy_Before_Defense": r["accuracy_before_defense"] / 100.0,
            "Accuracy_After_Defense": r["accuracy_after_defense"] / 100.0,
        }
        for r in defense_results
    ])
    def_df.to_csv(cfg.METRICS_DIR / "defense_results.csv", index=False)
    print("Exported CSV summaries to metrics directory.")

    return full_payload


if __name__ == "__main__":
    run_full_evaluation()
