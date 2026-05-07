"""Adversarial robustness evaluation for MobileNetV2, ResNet50, and ensemble."""

import gc
from pathlib import Path
from typing import Callable, Dict

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
CW_C = 0.1
CW_STEPS = 10
CW_LR = 0.01
CW_KAPPA = 0.0
EXAMPLE_COUNT = 3


def _clear_memory() -> None:
    """Release TensorFlow graph/session memory aggressively."""
    tf.keras.backend.clear_session()
    gc.collect()


def _preprocess_for(model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    """Apply ImageNet preprocess function to [0,1] images."""
    images_255 = images_01 * 255.0
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(images_255)
    if model_name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input(images_255)
    raise ValueError(f"Unsupported preprocess model: {model_name}")


def _predict_probs(model: tf.keras.Model, model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    inputs = _preprocess_for(model_name, images_01)
    return model(inputs, training=False)


def _ensemble_probs(
    mobilenet: tf.keras.Model,
    resnet: tf.keras.Model,
    images_01: tf.Tensor,
) -> tf.Tensor:
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
    """Single-step FGSM on [0,1] image space."""
    with tf.GradientTape() as tape:
        tape.watch(x)
        probs = probs_fn(x)
        loss = tf.keras.losses.categorical_crossentropy(y, probs)
        loss = tf.reduce_mean(loss)
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
    """Iterative PGD (L-infinity) on [0,1] image space."""
    x_orig = tf.identity(x)
    x_adv = tf.identity(x)

    for step in range(steps):
        print(f"  PGD step {step + 1}/{steps}")
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            probs = probs_fn(x_adv)
            loss = tf.keras.losses.categorical_crossentropy(y, probs)
            loss = tf.reduce_mean(loss)
        grad = tape.gradient(loss, x_adv)
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x_orig - eps, x_orig + eps)
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

    return x_adv


def _cw_simplified_attack(
    x: tf.Tensor,
    y: tf.Tensor,
    probs_fn: Callable[[tf.Tensor], tf.Tensor],
    c: float = CW_C,
    steps: int = CW_STEPS,
    lr: float = CW_LR,
    kappa: float = CW_KAPPA,
) -> tf.Tensor:
    """Lightweight simplified CW-style untargeted attack for CPU usage."""
    x_orig = tf.identity(x)
    x_adv = tf.Variable(x)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for step in range(steps):
        print(f"  CW step {step + 1}/{steps}")
        with tf.GradientTape() as tape:
            probs = probs_fn(x_adv)
            true_prob = tf.reduce_sum(y * probs, axis=1)
            other_prob = tf.reduce_max((1.0 - y) * probs, axis=1)
            # Encourage other class confidence >= true class confidence.
            f_loss = tf.nn.relu(true_prob - other_prob + kappa)
            l2_loss = tf.reduce_sum(tf.square(x_adv - x_orig), axis=[1, 2, 3])
            loss = tf.reduce_mean(l2_loss + c * f_loss)

        grads = tape.gradient(loss, x_adv)
        optimizer.apply_gradients([(grads, x_adv)])
        x_adv.assign(tf.clip_by_value(x_adv, 0.0, 1.0))

    return tf.convert_to_tensor(x_adv)


def _save_example_images(
    output_dir: Path,
    model_tag: str,
    x_orig: tf.Tensor,
    x_fgsm: tf.Tensor,
    x_pgd: tf.Tensor,
    x_cw: tf.Tensor,
    count: int = EXAMPLE_COUNT,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(count, int(x_orig.shape[0]))

    for idx in range(n):
        plt.imsave(output_dir / f"{model_tag}_sample{idx}_original.png", x_orig[idx])
        plt.imsave(output_dir / f"{model_tag}_sample{idx}_fgsm.png", x_fgsm[idx])
        plt.imsave(output_dir / f"{model_tag}_sample{idx}_pgd.png", x_pgd[idx])
        plt.imsave(output_dir / f"{model_tag}_sample{idx}_cw.png", x_cw[idx])


def _run_attacks_for_target(
    name: str,
    probs_fn: Callable[[tf.Tensor], tf.Tensor],
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
) -> Dict[str, float]:
    print(f"\n=== Evaluating {name} ===")
    print(f"[{name}] Sample progress: processing {len(x_test_np)} samples")
    x_test = tf.convert_to_tensor(x_test_np, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test_np, dtype=tf.float32)

    clean_probs = probs_fn(x_test)
    clean_acc = _accuracy_from_probs(clean_probs, y_test)
    print(f"[{name}] Clean accuracy: {clean_acc:.4f}")

    print(f"[{name}] Running FGSM...")
    x_fgsm = _fgsm_attack(x_test, y_test, probs_fn=probs_fn, eps=FGSM_EPS)
    fgsm_probs = probs_fn(x_fgsm)
    fgsm_acc = _accuracy_from_probs(fgsm_probs, y_test)
    print(f"[{name}] FGSM accuracy: {fgsm_acc:.4f}")

    print(f"[{name}] Running PGD...")
    x_pgd = _pgd_attack(x_test, y_test, probs_fn=probs_fn, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
    pgd_probs = probs_fn(x_pgd)
    pgd_acc = _accuracy_from_probs(pgd_probs, y_test)
    print(f"[{name}] PGD accuracy: {pgd_acc:.4f}")

    print(f"[{name}] Running simplified CW...")
    x_cw = _cw_simplified_attack(x_test, y_test, probs_fn=probs_fn, c=CW_C, steps=CW_STEPS, lr=CW_LR, kappa=CW_KAPPA)
    cw_probs = probs_fn(x_cw)
    cw_acc = _accuracy_from_probs(cw_probs, y_test)
    print(f"[{name}] CW accuracy: {cw_acc:.4f}")

    _save_example_images(
        output_dir=cfg.ADVERSARIAL_EXAMPLES_DIR,
        model_tag=name.lower().replace(" ", "_"),
        x_orig=x_test_np,
        x_fgsm=x_fgsm.numpy(),
        x_pgd=x_pgd.numpy(),
        x_cw=x_cw.numpy(),
    )

    del x_test, y_test, clean_probs, fgsm_probs, pgd_probs, cw_probs, x_fgsm, x_pgd, x_cw

    return {
        "Model": name,
        "Clean_Accuracy": clean_acc,
        "FGSM_Accuracy": fgsm_acc,
        "PGD_Accuracy": pgd_acc,
        "CW_Accuracy": cw_acc,
    }


def main() -> None:
    cfg.ensure_project_dirs()
    results_path = cfg.METRICS_DIR / "adversarial_results.csv"
    columns = ["Model", "Clean_Accuracy", "FGSM_Accuracy", "PGD_Accuracy", "CW_Accuracy"]
    rows = []

    print("Loading test subset...")
    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name=None)
    subset_ds = data_dict["test_ds"].unbatch().take(MAX_SAMPLES).batch(MAX_SAMPLES)
    x_test, y_test = next(iter(subset_ds))
    x_test_np = tf.cast(x_test, tf.float32).numpy()
    y_test_np = tf.cast(y_test, tf.float32).numpy()
    print(f"Using {x_test_np.shape[0]} test samples for adversarial evaluation.")
    del data_dict, subset_ds, x_test, y_test

    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"

    try:
        print(f"\nLoading model for MobileNetV2: {mobilenet_path}")
        mobilenet = tf.keras.models.load_model(mobilenet_path)
        mobilenet_probs_fn = lambda x: _predict_probs(mobilenet, "mobilenetv2", x)
        rows.append(_run_attacks_for_target("MobileNetV2", mobilenet_probs_fn, x_test_np, y_test_np))
        pd.DataFrame(rows, columns=columns).to_csv(results_path, index=False)
        del mobilenet, mobilenet_probs_fn
    except Exception as exc:
        print(f"[ERROR] MobileNetV2 evaluation failed: {exc}")
    finally:
        _clear_memory()

    try:
        print(f"\nLoading model for ResNet50: {resnet_path}")
        resnet = tf.keras.models.load_model(resnet_path)
        resnet_probs_fn = lambda x: _predict_probs(resnet, "resnet50", x)
        rows.append(_run_attacks_for_target("ResNet50", resnet_probs_fn, x_test_np, y_test_np))
        pd.DataFrame(rows, columns=columns).to_csv(results_path, index=False)
        del resnet, resnet_probs_fn
    except Exception as exc:
        print(f"[ERROR] ResNet50 evaluation failed: {exc}")
    finally:
        _clear_memory()

    try:
        print(f"\nLoading models for Soft Voting Ensemble: {mobilenet_path}, {resnet_path}")
        mobilenet = tf.keras.models.load_model(mobilenet_path)
        resnet = tf.keras.models.load_model(resnet_path)
        ensemble_probs_fn = lambda x: _ensemble_probs(mobilenet, resnet, x)
        rows.append(_run_attacks_for_target("Soft Voting Ensemble", ensemble_probs_fn, x_test_np, y_test_np))
        pd.DataFrame(rows, columns=columns).to_csv(results_path, index=False)
        del mobilenet, resnet, ensemble_probs_fn
    except Exception as exc:
        print(f"[ERROR] Soft Voting Ensemble evaluation failed: {exc}")
    finally:
        _clear_memory()

    results_df = pd.DataFrame(rows, columns=columns)
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved adversarial comparison table: {results_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
