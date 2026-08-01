"""Training module for MobileNetV2, EfficientNetB0, and ResNet50 on HAM10000."""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config as cfg
from data_loader import load_ham10000_data
from robust_skin_net import build_rasc_net
from run_attacks import _fgsm_attack


def get_curriculum_fgsm_ratio(
    current_epoch: int,
    total_epochs: int,
    curriculum: list = None,
) -> float:
    """Calculate dynamic FGSM ratio based on percentage completion of total epochs.
    
    Curriculum tuples: (progress_fraction, fgsm_ratio)
    e.g. [(0.25, 0.0), (0.50, 0.25), (1.00, 0.50)]
    Automatically adapts if total_epochs changes (percentage-based, no hardcoded epoch numbers).
    """
    if not getattr(cfg, "ENABLE_ADVERSARIAL_TRAINING", True):
        return 0.0

    if curriculum is None:
        curriculum = getattr(
            cfg,
            "ADVERSARIAL_CURRICULUM",
            [(0.25, 0.00), (0.50, 0.25), (1.00, 0.50)],
        )

    progress = (current_epoch + 1) / float(total_epochs)
    for max_prog, fgsm_ratio in curriculum:
        if progress <= max_prog + 1e-6:
            return fgsm_ratio
    return curriculum[-1][1]


def apply_adversarial_curriculum_to_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    sample_weights: tf.Tensor,
    model: tf.keras.Model,
    fgsm_ratio: float,
    epsilon: float = None,
) -> tuple:
    """Inject FGSM adversarial images into a training batch based on curriculum ratio."""
    if fgsm_ratio <= 0.0 or model is None:
        return images, labels, sample_weights

    if epsilon is None:
        epsilon = getattr(cfg, "FGSM_EPSILON", 0.01)

    batch_size = tf.shape(images)[0]
    fgsm_count = tf.cast(tf.cast(batch_size, tf.float32) * fgsm_ratio, tf.int32)

    if fgsm_count <= 0:
        return images, labels, sample_weights

    clean_images = images[:-fgsm_count]
    clean_labels = labels[:-fgsm_count]
    clean_weights = sample_weights[:-fgsm_count]

    target_images = images[-fgsm_count:]
    target_labels = labels[-fgsm_count:]
    target_weights = sample_weights[-fgsm_count:]

    probs_fn = lambda img: model(img, training=True)
    adv_images = _fgsm_attack(target_images, target_labels, probs_fn=probs_fn, eps=epsilon)


    final_images = tf.concat([clean_images, adv_images], axis=0)
    final_labels = tf.concat([clean_labels, target_labels], axis=0)
    final_weights = tf.concat([clean_weights, target_weights], axis=0)

    return final_images, final_labels, final_weights


def _get_training_constants() -> Tuple[int, int, int, float]:

    """Read training constants from config with safe defaults."""
    epochs_initial = getattr(cfg, "EPOCHS_INITIAL", 8)
    epochs_fine_tune = getattr(cfg, "EPOCHS_FINE_TUNE", 5)
    fine_tune_layers = getattr(cfg, "FINE_TUNE_LAYERS", 30)
    learning_rate = getattr(cfg, "LEARNING_RATE", 1e-4)
    return epochs_initial, epochs_fine_tune, fine_tune_layers, learning_rate


def _one_hot_and_weighted_train_ds(
    train_ds: tf.data.Dataset,
    class_weights: Dict[int, float],
    num_classes: int,
    enable_mixup: bool = None,
    mixup_alpha: float = None,
) -> tf.data.Dataset:
    """Attach sample weights using class weights and optionally apply MixUp augmentation."""
    if enable_mixup is None:
        enable_mixup = getattr(cfg, "ENABLE_MIXUP", False)
    if mixup_alpha is None:
        mixup_alpha = getattr(cfg, "MIXUP_ALPHA", 0.2)

    class_weight_values = tf.constant(
        [class_weights[idx] for idx in range(num_classes)],
        dtype=tf.float32,
    )

    def _map_fn(images: tf.Tensor, labels: tf.Tensor):
        sparse_labels = tf.argmax(labels, axis=-1, output_type=tf.int32)
        sample_weights = tf.gather(class_weight_values, sparse_labels)

        if enable_mixup:
            batch_size = tf.shape(images)[0]
            # Viva Mathematical Note:
            # If G1 ~ Gamma(alpha, 1) and G2 ~ Gamma(alpha, 1) are independent random variables,
            # X = G1 / (G1 + G2) follows a Beta(alpha, alpha) distribution.
            # Generating lambda via tf.random.gamma provides native TensorFlow GPU graph execution.
            gamma1 = tf.random.gamma([batch_size, 1, 1, 1], mixup_alpha, 1.0)
            gamma2 = tf.random.gamma([batch_size, 1, 1, 1], mixup_alpha, 1.0)
            lam_img = gamma1 / (gamma1 + gamma2 + 1e-8)

            shuffled_idx = tf.random.shuffle(tf.range(batch_size))
            images_shuffled = tf.gather(images, shuffled_idx)
            labels_shuffled = tf.gather(labels, shuffled_idx)
            weights_shuffled = tf.gather(sample_weights, shuffled_idx)

            mixed_images = lam_img * images + (1.0 - lam_img) * images_shuffled
            
            lam_label = tf.reshape(lam_img, [batch_size, 1])
            mixed_labels = lam_label * labels + (1.0 - lam_label) * labels_shuffled

            lam_weight = tf.reshape(lam_img, [batch_size])
            mixed_weights = lam_weight * sample_weights + (1.0 - lam_weight) * weights_shuffled

            return mixed_images, mixed_labels, mixed_weights

        return images, labels, sample_weights

    return train_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)




def _build_transfer_classifier(
    num_classes: int,
    backbone_name: str,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Build transfer-learning classifier with frozen backbone base."""
    if backbone_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        model_name = "mobilenetv2_ham10000"
    elif backbone_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        model_name = "efficientnetb0_ham10000"
    elif backbone_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        model_name = "resnet50_ham10000"
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model, base_model


def _categorical_focal_loss(gamma: float, label_smoothing: float = 0.0):
    """Create categorical focal loss callable for one-hot multiclass targets."""

    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        ce = tf.keras.losses.categorical_crossentropy(
            y_true,
            y_pred,
            label_smoothing=label_smoothing,
        )
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_factor = tf.pow(1.0 - p_t, gamma)
        return focal_factor * ce

    return loss_fn


class ValidationMacroF1Callback(tf.keras.callbacks.Callback):
    """Compute and print validation macro F1 at each epoch end."""

    def __init__(self, val_ds: tf.data.Dataset, val_true_labels: np.ndarray) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.val_true_labels = val_true_labels.astype(int)

    def on_epoch_end(self, epoch, logs=None) -> None:
        y_prob = self.model.predict(self.val_ds, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        macro_f1 = f1_score(self.val_true_labels, y_pred, average="macro", zero_division=0)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
        print(f"\nEpoch {epoch + 1}: val_macro_f1={macro_f1:.4f}")


def get_standard_callbacks(best_checkpoint_path: str) -> list:
    """Construct standard callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau."""
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )
    return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]


def log_and_save_experiment_manifest(
    model: tf.keras.Model,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    manifest_path: Path,
) -> dict:
    """Print configuration summary and save JSON experiment manifest."""
    trainable_params = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    total_params = int(model.count_params())

    manifest = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "optimizer": "Adam",
        "initial_learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "mixup_enabled": bool(getattr(cfg, "ENABLE_MIXUP", True)),
        "mixup_alpha": float(getattr(cfg, "MIXUP_ALPHA", 0.2)),
        "label_smoothing_enabled": bool(getattr(cfg, "ENABLE_LABEL_SMOOTHING", True)),
        "label_smoothing_value": float(getattr(cfg, "LABEL_SMOOTHING", 0.1)),
        "adversarial_training_enabled": bool(getattr(cfg, "ENABLE_ADVERSARIAL_TRAINING", True)),
        "fgsm_epsilon": float(getattr(cfg, "FGSM_EPSILON", 0.01)),
        "adversarial_curriculum": getattr(cfg, "ADVERSARIAL_CURRICULUM", []),
        "class_weights_enabled": True,
    }

    print("\n" + "=" * 60)
    print(f"[CONFIG SUMMARY] TRAINING CONFIGURATION: {model_name}")
    print("=" * 60)
    for key, value in manifest.items():
        print(f"  * {key.replace('_', ' ').title()}: {value}")
    print("=" * 60 + "\n")


    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def save_training_history_csv(history: tf.keras.callbacks.History, csv_path: Path) -> None:
    """Save epoch-by-epoch training metrics to CSV file."""
    import pandas as pd

    hist_dict = history.history
    epochs_count = len(hist_dict.get("loss", []))
    lrs = hist_dict.get("lr", hist_dict.get("learning_rate", [cfg.LEARNING_RATE] * epochs_count))

    df = pd.DataFrame({
        "epoch": list(range(1, epochs_count + 1)),
        "train_loss": hist_dict.get("loss", []),
        "val_loss": hist_dict.get("val_loss", []),
        "train_accuracy": hist_dict.get("accuracy", hist_dict.get("categorical_accuracy", [])),
        "val_accuracy": hist_dict.get("val_accuracy", []),
        "learning_rate": lrs,
    })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Training history saved to: {csv_path}")


def _compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    loss_fn=None,
    label_smoothing: float = None,
    enable_label_smoothing: bool = None,
) -> None:
    """Compile the model with Adam + configurable label-smoothed categorical crossentropy."""
    if enable_label_smoothing is None:
        enable_label_smoothing = getattr(cfg, "ENABLE_LABEL_SMOOTHING", True)
    if label_smoothing is None:
        label_smoothing = getattr(cfg, "LABEL_SMOOTHING", 0.1)

    effective_smoothing = label_smoothing if enable_label_smoothing else 0.0

    if loss_fn is None:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=effective_smoothing)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )




def _plot_training_curves(
    history_initial,
    history_finetune,
    save_path: Path,
    model_display_name: str,
) -> None:
    """Save combined accuracy/loss curves for both training phases."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history_initial.history.get("accuracy", []) + history_finetune.history.get("accuracy", [])
    val_acc = history_initial.history.get("val_accuracy", []) + history_finetune.history.get("val_accuracy", [])
    loss = history_initial.history.get("loss", []) + history_finetune.history.get("loss", [])
    val_loss = history_initial.history.get("val_loss", []) + history_finetune.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.title(f"{model_display_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title(f"{model_display_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_test_metrics(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_df,
    index_to_label: Dict[int, str],
    report_path: Path,
    confusion_plot_path: Path,
    model_display_name: str,
) -> None:
    """Compute and save test accuracy, classification report, and confusion matrix plot."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_plot_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = test_df["label"].values.astype(int)
    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    test_accuracy = float(np.mean(y_true == y_pred))

    label_indices = sorted(index_to_label.keys())
    target_names = [index_to_label[idx] for idx in label_indices]
    report = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_indices)

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{model_display_name} Test Metrics\n")
        handle.write("=" * 28 + "\n")
        handle.write(f"Accuracy: {test_accuracy:.4f}\n\n")
        handle.write("Classification Report:\n")
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
    plt.title(f"{model_display_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(confusion_plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def _save_class_mapping(mapping_path: Path, label_to_index: Dict[str, int], index_to_label: Dict[int, str]) -> None:
    """Persist class mappings used during training for later aligned inference."""
    payload = {
        "label_to_index": {str(label): int(idx) for label, idx in label_to_index.items()},
        "index_to_label": {str(int(idx)): str(label) for idx, label in index_to_label.items()},
    }
    with mapping_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_mobilenetv2() -> Dict[str, object]:
    """Train MobileNetV2 in two phases and export metrics/plots/artifacts."""
    cfg.ensure_project_dirs()
    epochs_initial, epochs_fine_tune, fine_tune_layers, learning_rate = _get_training_constants()

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name="mobilenetv2")
    num_classes = len(data_dict["label_to_index"])

    train_ds = _one_hot_and_weighted_train_ds(
        train_ds=data_dict["train_ds"],
        class_weights=data_dict["class_weights"],
        num_classes=num_classes,
    )
    val_ds = data_dict["val_ds"]
    test_ds_eval = data_dict["test_ds"]
    test_ds_raw = data_dict["test_ds"]

    model, base_model = _build_transfer_classifier(
        num_classes=num_classes,
        backbone_name="MobileNetV2",
    )
    _compile_model(model, learning_rate=learning_rate)

    best_checkpoint_path = cfg.MODELS_DIR / "mobilenetv2_best.keras"
    final_model_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    class_mapping_path = cfg.MODELS_DIR / "mobilenetv2_class_mapping.json"
    training_plot_path = cfg.PLOTS_DIR / "mobilenetv2_training.png"
    report_path = cfg.METRICS_DIR / "mobilenetv2_report.txt"
    confusion_plot_path = cfg.PLOTS_DIR / "mobilenetv2_confusion_matrix.png"

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print("Starting initial training phase (frozen MobileNetV2 base)...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    print("Starting fine-tuning phase...")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    _compile_model(model, learning_rate=1e-5)
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial + epochs_fine_tune,
        initial_epoch=epochs_initial,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    model.save(final_model_path)
    _save_class_mapping(
        mapping_path=class_mapping_path,
        label_to_index=data_dict["label_to_index"],
        index_to_label=data_dict["index_to_label"],
    )
    _plot_training_curves(
        history_initial,
        history_finetune,
        training_plot_path,
        model_display_name="MobileNetV2",
    )

    test_loss, test_acc = model.evaluate(test_ds_eval, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    _save_test_metrics(
        model=model,
        test_ds=test_ds_raw,
        test_df=data_dict["test_df"],
        index_to_label=data_dict["index_to_label"],
        report_path=report_path,
        confusion_plot_path=confusion_plot_path,
        model_display_name="MobileNetV2",
    )

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_model_path": str(final_model_path),
        "class_mapping_path": str(class_mapping_path),
        "training_plot_path": str(training_plot_path),
        "report_path": str(report_path),
        "confusion_plot_path": str(confusion_plot_path),
    }


def train_efficientnetb0() -> Dict[str, object]:
    """Train EfficientNetB0 in two phases and export metrics/plots/artifacts."""
    cfg.ensure_project_dirs()
    epochs_initial, epochs_fine_tune, fine_tune_layers, learning_rate = _get_training_constants()

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name="efficientnetb0")
    num_classes = len(data_dict["label_to_index"])
    val_true_labels = data_dict["val_df"]["label"].values.astype(int)

    train_ds = _one_hot_and_weighted_train_ds(
        train_ds=data_dict["train_ds"],
        class_weights=data_dict["class_weights"],
        num_classes=num_classes,
    )
    val_ds = data_dict["val_ds"]
    test_ds_eval = data_dict["test_ds"]
    test_ds_raw = data_dict["test_ds"]

    model, base_model = _build_transfer_classifier(
        num_classes=num_classes,
        backbone_name="EfficientNetB0",
    )
    use_focal_loss = getattr(cfg, "USE_FOCAL_LOSS", True)
    focal_gamma = getattr(cfg, "FOCAL_GAMMA", 2.0)
    label_smoothing = getattr(cfg, "LABEL_SMOOTHING", 0.1)

    if use_focal_loss:
        loss_fn = _categorical_focal_loss(gamma=focal_gamma, label_smoothing=label_smoothing)
        print(f"Using focal loss for EfficientNetB0 (gamma={focal_gamma}, label_smoothing={label_smoothing})")
    else:
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        print(f"Using categorical crossentropy for EfficientNetB0 (label_smoothing={label_smoothing})")

    _compile_model(model, learning_rate=learning_rate, loss_fn=loss_fn)

    best_checkpoint_path = cfg.MODELS_DIR / "efficientnetb0_best.keras"
    final_model_path = cfg.MODELS_DIR / "efficientnetb0_finetuned.keras"
    class_mapping_path = cfg.MODELS_DIR / "efficientnetb0_class_mapping.json"
    training_plot_path = cfg.PLOTS_DIR / "efficientnetb0_training.png"
    report_path = cfg.METRICS_DIR / "efficientnetb0_report.txt"
    confusion_plot_path = cfg.PLOTS_DIR / "efficientnetb0_confusion_matrix.png"

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    macro_f1_cb = ValidationMacroF1Callback(val_ds=val_ds, val_true_labels=val_true_labels)

    print("Starting initial training phase (frozen EfficientNetB0 base)...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial,
        callbacks=[checkpoint_cb, macro_f1_cb],
        verbose=1,
    )

    print("Starting fine-tuning phase...")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    _compile_model(model, learning_rate=1e-5, loss_fn=loss_fn)
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial + epochs_fine_tune,
        initial_epoch=epochs_initial,
        callbacks=[checkpoint_cb, macro_f1_cb],
        verbose=1,
    )

    model.save(final_model_path)
    _save_class_mapping(
        mapping_path=class_mapping_path,
        label_to_index=data_dict["label_to_index"],
        index_to_label=data_dict["index_to_label"],
    )
    _plot_training_curves(
        history_initial,
        history_finetune,
        training_plot_path,
        model_display_name="EfficientNetB0",
    )

    test_loss, test_acc = model.evaluate(test_ds_eval, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    _save_test_metrics(
        model=model,
        test_ds=test_ds_raw,
        test_df=data_dict["test_df"],
        index_to_label=data_dict["index_to_label"],
        report_path=report_path,
        confusion_plot_path=confusion_plot_path,
        model_display_name="EfficientNetB0",
    )

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_model_path": str(final_model_path),
        "class_mapping_path": str(class_mapping_path),
        "training_plot_path": str(training_plot_path),
        "report_path": str(report_path),
        "confusion_plot_path": str(confusion_plot_path),
    }


def train_resnet50() -> Dict[str, object]:
    """Train ResNet50 in two phases and export metrics/plots/artifacts."""
    cfg.ensure_project_dirs()
    epochs_initial, epochs_fine_tune, fine_tune_layers, learning_rate = _get_training_constants()

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name="resnet50")
    num_classes = len(data_dict["label_to_index"])

    train_ds = _one_hot_and_weighted_train_ds(
        train_ds=data_dict["train_ds"],
        class_weights=data_dict["class_weights"],
        num_classes=num_classes,
    )
    val_ds = data_dict["val_ds"]
    test_ds_eval = data_dict["test_ds"]
    test_ds_raw = data_dict["test_ds"]

    model, base_model = _build_transfer_classifier(
        num_classes=num_classes,
        backbone_name="ResNet50",
    )
    _compile_model(model, learning_rate=learning_rate)

    best_checkpoint_path = cfg.MODELS_DIR / "resnet50_best.keras"
    final_model_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    class_mapping_path = cfg.MODELS_DIR / "resnet50_class_mapping.json"
    training_plot_path = cfg.PLOTS_DIR / "resnet50_training.png"
    report_path = cfg.METRICS_DIR / "resnet50_report.txt"
    confusion_plot_path = cfg.PLOTS_DIR / "resnet50_confusion_matrix.png"

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_checkpoint_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    print("Starting initial training phase (frozen ResNet50 base)...")
    history_initial = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    print("Starting fine-tuning phase...")
    base_model.trainable = True
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    _compile_model(model, learning_rate=1e-5)
    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_initial + epochs_fine_tune,
        initial_epoch=epochs_initial,
        callbacks=[checkpoint_cb],
        verbose=1,
    )

    model.save(final_model_path)
    _save_class_mapping(
        mapping_path=class_mapping_path,
        label_to_index=data_dict["label_to_index"],
        index_to_label=data_dict["index_to_label"],
    )
    _plot_training_curves(
        history_initial,
        history_finetune,
        training_plot_path,
        model_display_name="ResNet50",
    )

    test_loss, test_acc = model.evaluate(test_ds_eval, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    _save_test_metrics(
        model=model,
        test_ds=test_ds_raw,
        test_df=data_dict["test_df"],
        index_to_label=data_dict["index_to_label"],
        report_path=report_path,
        confusion_plot_path=confusion_plot_path,
        model_display_name="ResNet50",
    )

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_model_path": str(final_model_path),
        "class_mapping_path": str(class_mapping_path),
        "training_plot_path": str(training_plot_path),
        "report_path": str(report_path),
        "confusion_plot_path": str(confusion_plot_path),
    }


def train_rasc_net() -> Dict[str, object]:
    """Train proposed custom RASC-Net model architecture on HAM10000."""
    cfg.ensure_project_dirs()
    epochs_initial, epochs_fine_tune, _, learning_rate = _get_training_constants()

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name=None)
    num_classes = len(data_dict["label_to_index"])

    train_ds = _one_hot_and_weighted_train_ds(
        train_ds=data_dict["train_ds"],
        class_weights=data_dict["class_weights"],
        num_classes=num_classes,
    )
    val_ds = data_dict["val_ds"]
    test_ds_raw = data_dict["test_ds"]

    print("Building custom RASC-Net architecture (from scratch)...")
    model = build_rasc_net(input_shape=(224, 224, 3), num_classes=num_classes)

    best_checkpoint_path = cfg.MODELS_DIR / "rasc_net_best.keras"
    final_model_path = cfg.MODELS_DIR / "rasc_net_finetuned.keras"
    manifest_path = cfg.MODELS_DIR / "rasc_net_manifest.json"
    history_csv_path = cfg.OUTPUTS_DIR / "metrics" / "rasc_net_history.csv"
    class_mapping_path = cfg.MODELS_DIR / "rasc_net_class_mapping.json"
    training_plot_path = cfg.PLOTS_DIR / "rasc_net_training.png"
    report_path = cfg.METRICS_DIR / "rasc_net_report.txt"
    confusion_plot_path = cfg.PLOTS_DIR / "rasc_net_confusion_matrix.png"

    total_epochs = epochs_initial + epochs_fine_tune
    callbacks = get_standard_callbacks(best_checkpoint_path)

    log_and_save_experiment_manifest(
        model=model,
        model_name="RASC-Net",
        epochs=total_epochs,
        batch_size=cfg.BATCH_SIZE,
        learning_rate=learning_rate,
        manifest_path=manifest_path,
    )

    print("Starting training for RASC-Net...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=total_epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_model_path)
    save_training_history_csv(history, history_csv_path)

    _save_class_mapping(
        mapping_path=class_mapping_path,
        label_to_index=data_dict["label_to_index"],
        index_to_label=data_dict["index_to_label"],
    )

    _save_test_metrics(
        model=model,
        test_ds=test_ds_raw,
        test_df=data_dict["test_df"],
        index_to_label=data_dict["index_to_label"],
        report_path=report_path,
        confusion_plot_path=confusion_plot_path,
        model_display_name="RASC-Net",
    )

    return {
        "model": model,
        "best_checkpoint_path": str(best_checkpoint_path),
        "final_model_path": str(final_model_path),
        "manifest_path": str(manifest_path),
        "history_csv_path": str(history_csv_path),
        "class_mapping_path": str(class_mapping_path),
        "training_plot_path": str(training_plot_path),
        "report_path": str(report_path),
        "confusion_plot_path": str(confusion_plot_path),
    }



if __name__ == "__main__":
    train_mobilenetv2()

