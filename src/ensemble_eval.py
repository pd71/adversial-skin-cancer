"""Evaluate soft-voting ensemble using trained MobileNetV2 and ResNet50 models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import config as cfg
from data_loader import load_ham10000_data


def _load_model_mapping(mapping_path: Path, fallback_index_to_label: dict, fallback_label_to_index: dict) -> dict:
    """Load saved mapping JSON if present, otherwise fallback to dataset mapping."""
    if not mapping_path.exists():
        return {
            "index_to_label": {int(k): v for k, v in fallback_index_to_label.items()},
            "label_to_index": {k: int(v) for k, v in fallback_label_to_index.items()},
        }

    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "index_to_label": {int(k): str(v) for k, v in payload["index_to_label"].items()},
        "label_to_index": {str(k): int(v) for k, v in payload["label_to_index"].items()},
    }


def _reorder_probs_to_reference(
    probs: np.ndarray,
    model_index_to_label: dict,
    reference_label_to_index: dict,
) -> np.ndarray:
    """Reorder model probability columns to match reference dataset class order."""
    reordered = np.zeros((probs.shape[0], len(reference_label_to_index)), dtype=probs.dtype)
    for model_idx, label in model_index_to_label.items():
        if label not in reference_label_to_index:
            raise ValueError(f"Label '{label}' not found in reference mapping.")
        ref_idx = reference_label_to_index[label]
        reordered[:, ref_idx] = probs[:, model_idx]
    return reordered


def evaluate_mobilenet_resnet_ensemble() -> None:
    """Load trained models, run test inference, and save ensemble metrics/plots."""
    cfg.ensure_project_dirs()

    mobilenet_model_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_model_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    mobilenet_mapping_path = cfg.MODELS_DIR / "mobilenetv2_class_mapping.json"
    resnet_mapping_path = cfg.MODELS_DIR / "resnet50_class_mapping.json"

    if not mobilenet_model_path.exists():
        raise FileNotFoundError(f"Missing model file: {mobilenet_model_path}")
    if not resnet_model_path.exists():
        raise FileNotFoundError(f"Missing model file: {resnet_model_path}")

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE)
    test_ds = data_dict["test_ds"]
    test_df = data_dict["test_df"]
    index_to_label = data_dict["index_to_label"]
    label_to_index = data_dict["label_to_index"]

    y_true = test_df["label"].values.astype(int)
    label_indices = sorted(index_to_label.keys())
    target_names = [index_to_label[idx] for idx in label_indices]

    mobilenet_model = tf.keras.models.load_model(mobilenet_model_path)
    resnet_model = tf.keras.models.load_model(resnet_model_path)

    mobilenet_probs = mobilenet_model.predict(test_ds, verbose=0)
    resnet_probs = resnet_model.predict(test_ds, verbose=0)
    mobilenet_mapping = _load_model_mapping(
        mapping_path=mobilenet_mapping_path,
        fallback_index_to_label=index_to_label,
        fallback_label_to_index=label_to_index,
    )
    resnet_mapping = _load_model_mapping(
        mapping_path=resnet_mapping_path,
        fallback_index_to_label=index_to_label,
        fallback_label_to_index=label_to_index,
    )

    mobilenet_probs = _reorder_probs_to_reference(
        probs=mobilenet_probs,
        model_index_to_label=mobilenet_mapping["index_to_label"],
        reference_label_to_index=label_to_index,
    )
    resnet_probs = _reorder_probs_to_reference(
        probs=resnet_probs,
        model_index_to_label=resnet_mapping["index_to_label"],
        reference_label_to_index=label_to_index,
    )
    ensemble_probs = (mobilenet_probs + resnet_probs) / 2.0

    mobilenet_pred = np.argmax(mobilenet_probs, axis=1)
    resnet_pred = np.argmax(resnet_probs, axis=1)
    ensemble_pred = np.argmax(ensemble_probs, axis=1)

    mobilenet_acc = float(np.mean(mobilenet_pred == y_true))
    resnet_acc = float(np.mean(resnet_pred == y_true))
    ensemble_acc = float(np.mean(ensemble_pred == y_true))

    report = classification_report(
        y_true,
        ensemble_pred,
        labels=label_indices,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, ensemble_pred, labels=label_indices)

    report_path = cfg.METRICS_DIR / "ensemble_mobilenet_resnet_report.txt"
    confusion_plot_path = cfg.PLOTS_DIR / "ensemble_mobilenet_resnet_confusion_matrix.png"

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("MobileNetV2 + ResNet50 Ensemble (Soft Voting)\n")
        handle.write("=" * 52 + "\n")
        handle.write(f"MobileNetV2 Accuracy: {mobilenet_acc:.4f}\n")
        handle.write(f"ResNet50 Accuracy:    {resnet_acc:.4f}\n")
        handle.write(f"Ensemble Accuracy:    {ensemble_acc:.4f}\n\n")
        handle.write("Classification Report (Ensemble):\n")
        handle.write(report)
        handle.write("\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Ensemble Confusion Matrix (MobileNetV2 + ResNet50)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(confusion_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("=== Ensemble Evaluation Complete ===")
    print(f"MobileNetV2 accuracy: {mobilenet_acc:.4f}")
    print(f"ResNet50 accuracy:    {resnet_acc:.4f}")
    print(f"Ensemble accuracy:    {ensemble_acc:.4f}")
    print(f"Saved report: {report_path}")
    print(f"Saved plot  : {confusion_plot_path}")


if __name__ == "__main__":
    evaluate_mobilenet_resnet_ensemble()
