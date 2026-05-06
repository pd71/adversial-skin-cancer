"""Evaluate simple defenses against FGSM/PGD adversarial attacks."""

from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
from data_loader import load_ham10000_data


MAX_SAMPLES = 50
FGSM_EPS = 0.03
PGD_EPS = 0.03
PGD_ALPHA = 0.007
PGD_STEPS = 5
EXAMPLE_COUNT = 3


def bit_depth_reduction(images: tf.Tensor, bits: int = 4) -> tf.Tensor:
    """Feature squeezing by quantizing image intensities."""
    levels = float(2**bits - 1)
    squeezed = tf.round(images * levels) / levels
    return tf.clip_by_value(squeezed, 0.0, 1.0)


def gaussian_blur(images: tf.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> tf.Tensor:
    """Apply lightweight gaussian blur using depthwise convolution."""
    coords = tf.range(kernel_size, dtype=tf.float32) - (kernel_size - 1) / 2.0
    g = tf.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / tf.reduce_sum(g)
    kernel_2d = tf.tensordot(g, g, axes=0)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel_2d, [1, 1, 3, 1])
    blurred = tf.nn.depthwise_conv2d(images, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return tf.clip_by_value(blurred, 0.0, 1.0)


def jpeg_compression_defense(images: tf.Tensor, jpeg_quality: int = 70) -> tf.Tensor:
    """Apply JPEG compression/decompression defense per image."""
    compressed = []
    for i in range(images.shape[0]):
        img = tf.cast(images[i] * 255.0, tf.uint8)
        jpeg_bytes = tf.io.encode_jpeg(img, quality=jpeg_quality)
        decoded = tf.io.decode_jpeg(jpeg_bytes, channels=3)
        decoded = tf.cast(decoded, tf.float32) / 255.0
        compressed.append(decoded)
    return tf.stack(compressed, axis=0)


def _defense_pipeline(images: tf.Tensor) -> tf.Tensor:
    """Apply all defenses sequentially."""
    x = bit_depth_reduction(images, bits=4)
    x = gaussian_blur(x, kernel_size=3, sigma=1.0)
    x = jpeg_compression_defense(x, jpeg_quality=70)
    return tf.clip_by_value(x, 0.0, 1.0)


def _preprocess_for(model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    images_255 = images_01 * 255.0
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(images_255)
    if model_name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input(images_255)
    raise ValueError(f"Unsupported preprocess model: {model_name}")


def _predict_probs(model: tf.keras.Model, model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    return model(_preprocess_for(model_name, images_01), training=False)


def _ensemble_probs(mobilenet: tf.keras.Model, resnet: tf.keras.Model, images_01: tf.Tensor) -> tf.Tensor:
    m_probs = _predict_probs(mobilenet, "mobilenetv2", images_01)
    r_probs = _predict_probs(resnet, "resnet50", images_01)
    return (m_probs + r_probs) / 2.0


def _accuracy_from_probs(probs: tf.Tensor, y_true_one_hot: tf.Tensor) -> float:
    y_true = tf.argmax(y_true_one_hot, axis=1).numpy()
    y_pred = tf.argmax(probs, axis=1).numpy()
    return float(np.mean(y_true == y_pred))


def _fgsm_attack(
    x: tf.Tensor,
    y: tf.Tensor,
    probs_fn: Callable[[tf.Tensor], tf.Tensor],
    eps: float = FGSM_EPS,
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = probs_fn(x)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, probs))
    grad = tape.gradient(loss, x)
    x_adv = x + eps * tf.sign(grad)
    return tf.clip_by_value(x_adv, 0.0, 1.0)


def _pgd_attack(
    x: tf.Tensor,
    y: tf.Tensor,
    probs_fn: Callable[[tf.Tensor], tf.Tensor],
    eps: float = PGD_EPS,
    alpha: float = PGD_ALPHA,
    steps: int = PGD_STEPS,
) -> tf.Tensor:
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


def _save_comparison_images(
    output_dir: Path,
    model_tag: str,
    attack_tag: str,
    x_orig: np.ndarray,
    x_adv: np.ndarray,
    x_def: np.ndarray,
    count: int = EXAMPLE_COUNT,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(count, x_orig.shape[0])
    for i in range(n):
        plt.imsave(output_dir / f"{model_tag}_{attack_tag}_sample{i}_original.png", x_orig[i])
        plt.imsave(output_dir / f"{model_tag}_{attack_tag}_sample{i}_adversarial.png", x_adv[i])
        plt.imsave(output_dir / f"{model_tag}_{attack_tag}_sample{i}_defended.png", x_def[i])


def _evaluate_model_attack(
    model_name: str,
    probs_fn: Callable[[tf.Tensor], tf.Tensor],
    attack_name: str,
    attack_fn: Callable[[tf.Tensor, tf.Tensor, Callable[[tf.Tensor], tf.Tensor]], tf.Tensor],
    x_clean: tf.Tensor,
    y_true: tf.Tensor,
) -> Dict[str, float]:
    print(f"[{model_name}] Running attack: {attack_name}")
    x_adv = attack_fn(x_clean, y_true, probs_fn)
    before_acc = _accuracy_from_probs(probs_fn(x_adv), y_true)

    x_def = _defense_pipeline(x_adv)
    after_acc = _accuracy_from_probs(probs_fn(x_def), y_true)

    _save_comparison_images(
        output_dir=cfg.ADVERSARIAL_EXAMPLES_DIR / "defended",
        model_tag=model_name.lower().replace(" ", "_"),
        attack_tag=attack_name.lower(),
        x_orig=x_clean.numpy(),
        x_adv=x_adv.numpy(),
        x_def=x_def.numpy(),
    )

    return {
        "Model": model_name,
        "Attack": attack_name,
        "Accuracy_Before_Defense": before_acc,
        "Accuracy_After_Defense": after_acc,
    }


def main() -> None:
    cfg.ensure_project_dirs()

    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    if not mobilenet_path.exists() or not resnet_path.exists():
        raise FileNotFoundError("Required model files are missing for defense evaluation.")

    print("Loading models...")
    mobilenet = tf.keras.models.load_model(mobilenet_path)
    resnet = tf.keras.models.load_model(resnet_path)

    print("Loading test subset...")
    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name=None)
    subset_ds = data_dict["test_ds"].unbatch().take(MAX_SAMPLES).batch(MAX_SAMPLES)
    x_test, y_test = next(iter(subset_ds))
    x_test = tf.cast(x_test, tf.float32)
    y_test = tf.cast(y_test, tf.float32)
    print(f"Using {x_test.shape[0]} samples.")

    attack_map = {
        "FGSM": _fgsm_attack,
        "PGD": _pgd_attack,
    }

    model_map = {
        "MobileNetV2": lambda x: _predict_probs(mobilenet, "mobilenetv2", x),
        "ResNet50": lambda x: _predict_probs(resnet, "resnet50", x),
        "Ensemble": lambda x: _ensemble_probs(mobilenet, resnet, x),
    }

    rows: List[Dict[str, float]] = []
    for model_name, probs_fn in model_map.items():
        print(f"\n=== Evaluating defenses for {model_name} ===")
        for attack_name, attack_fn in attack_map.items():
            row = _evaluate_model_attack(
                model_name=model_name,
                probs_fn=probs_fn,
                attack_name=attack_name,
                attack_fn=attack_fn,
                x_clean=x_test,
                y_true=y_test,
            )
            rows.append(row)
            print(
                f"[{model_name} | {attack_name}] "
                f"before={row['Accuracy_Before_Defense']:.4f}, "
                f"after={row['Accuracy_After_Defense']:.4f}"
            )

    results_df = pd.DataFrame(
        rows,
        columns=["Model", "Attack", "Accuracy_Before_Defense", "Accuracy_After_Defense"],
    )
    results_path = cfg.METRICS_DIR / "defense_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved defense results: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
