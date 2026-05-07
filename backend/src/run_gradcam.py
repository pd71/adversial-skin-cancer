"""Generate Grad-CAM visualizations for MobileNetV2 and ResNet50."""

from pathlib import Path
from typing import Callable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import config as cfg
from data_loader import load_ham10000_data


NUM_EXAMPLES_PER_MODEL = 5


def _preprocess_for(model_name: str, images_01: tf.Tensor) -> tf.Tensor:
    """Apply model-specific ImageNet preprocessing."""
    images_255 = images_01 * 255.0
    if model_name == "mobilenetv2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(images_255)
    if model_name == "resnet50":
        return tf.keras.applications.resnet.preprocess_input(images_255)
    raise ValueError(f"Unsupported model_name: {model_name}")


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Find last Conv2D layer, including nested backbone models."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not find a Conv2D layer for Grad-CAM.")


def generate_gradcam(
    model: tf.keras.Model,
    model_name: str,
    image_tensor: tf.Tensor,
    preprocess_fn: Callable[[tf.Tensor], tf.Tensor],
    last_conv_layer_name: Optional[str] = None,
) -> tuple[np.ndarray, int, float]:
    """Generate Grad-CAM heatmap for a single image tensor in [0,1]."""
    try:
        if last_conv_layer_name is None:
            last_conv_layer_name = find_last_conv_layer(model)

        image_tensor = tf.cast(image_tensor, tf.float32)
        if image_tensor.shape.rank == 3:
            image_tensor = tf.expand_dims(image_tensor, axis=0)
        image_tensor = tf.ensure_shape(image_tensor, [1, 224, 224, 3])
        print(f"[GradCAM] Model: {model_name} | Conv layer: {last_conv_layer_name} | Image shape: {image_tensor.shape}")

        preprocessed = preprocess_fn(image_tensor)

        # Handle nested Functional models safely
        inner_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    layer.get_layer(last_conv_layer_name)
                    inner_model = layer
                    break
                except ValueError:
                    pass

        with tf.GradientTape() as tape:
            tape.watch(preprocessed)
            conv_outputs, predictions = model(preprocessed)

        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if float(max_val.numpy()) > 0:
            heatmap = heatmap / max_val

        confidence = float(predictions[0][pred_index].numpy())
        return heatmap.numpy(), int(pred_index.numpy()), confidence
    except Exception as e:
        print(f"[GradCAM] Error generating Grad-CAM for {model_name}: {e}")
        raise


def _create_overlay(image_01: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    """Create colorized heatmap and overlay image."""
    h, w = image_01.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    image_uint8 = np.uint8(np.clip(image_01, 0.0, 1.0) * 255.0)
    overlay = cv2.addWeighted(image_uint8, 1.0 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, overlay


def _save_gradcam_figure(
    save_path: Path,
    original: np.ndarray,
    heatmap_color: np.ndarray,
    overlay: np.ndarray,
    title: str,
) -> None:
    """Save three-panel Grad-CAM visualization."""
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap_color)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM to: {save_path}")


def build_grad_model(model: tf.keras.Model, last_conv_layer_name: str) -> tf.keras.Model:
    inner_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                layer.get_layer(last_conv_layer_name)
                inner_model = layer
                break
            except ValueError:
                pass
                
    if inner_model:
        conv_layer = inner_model.get_layer(last_conv_layer_name)
        inner_grad_model = tf.keras.models.Model(
            inner_model.inputs,
            [conv_layer.output, inner_model.output]
        )
        new_inputs = tf.keras.Input(shape=(224, 224, 3))
        conv_outputs, x = inner_grad_model(new_inputs)
        idx = model.layers.index(inner_model)
        for layer in model.layers[idx+1:]:
            x = layer(x)
        return tf.keras.models.Model(new_inputs, [conv_outputs, x])
    else:
        conv_layer = model.get_layer(last_conv_layer_name)
        return tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

def main() -> None:
    cfg.ensure_project_dirs()
    gradcam_dir = (cfg.PROJECT_ROOT / "backend" / "outputs" / "plots" / "gradcam").resolve()
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    mobilenet_path = cfg.MODELS_DIR / "mobilenetv2_finetuned.keras"
    resnet_path = cfg.MODELS_DIR / "resnet50_finetuned.keras"
    if not mobilenet_path.exists() or not resnet_path.exists():
        raise FileNotFoundError("Required model files not found for Grad-CAM generation.")

    print("Loading models...")
    mobilenet = tf.keras.models.load_model(mobilenet_path)
    resnet = tf.keras.models.load_model(resnet_path)
    mobilenet_last_conv = find_last_conv_layer(mobilenet)
    resnet_last_conv = find_last_conv_layer(resnet)
    print(f"Using Grad-CAM layer: {mobilenet_last_conv} (MobileNetV2)")
    print(f"Using Grad-CAM layer: {resnet_last_conv} (ResNet50)")

    print("Loading test images...")
    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE, model_name=None)
    index_to_label = data_dict["index_to_label"]
    sample_ds = data_dict["test_ds"].unbatch().take(NUM_EXAMPLES_PER_MODEL)
    samples = list(sample_ds.as_numpy_iterator())

    print("Building grad models...")
    mobilenet_grad = build_grad_model(mobilenet, mobilenet_last_conv)
    resnet_grad = build_grad_model(resnet, resnet_last_conv)

    print("Generating MobileNetV2 Grad-CAM examples...")
    for idx, (image_np, _) in enumerate(samples):
        try:
            image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
            heatmap, pred_idx, conf = generate_gradcam(
                model=mobilenet_grad,
                model_name="MobileNetV2",
                image_tensor=image_tf,
                preprocess_fn=lambda x: _preprocess_for("mobilenetv2", x),
                last_conv_layer_name=mobilenet_last_conv,
            )
            heatmap_color, overlay = _create_overlay(image_np, heatmap)
            pred_label = index_to_label.get(pred_idx, str(pred_idx))
            title = f"MobileNetV2 | Pred: {pred_label} | Conf: {conf:.4f}"
            save_path = gradcam_dir / f"mobilenetv2_gradcam_{idx}.png"
            _save_gradcam_figure(save_path, image_np, heatmap_color, overlay, title)
        except Exception as exc:
            print(f"[GradCAM] Skipping MobileNetV2 image {idx} due to error: {exc}")
            continue

    print("Generating ResNet50 Grad-CAM examples...")
    for idx, (image_np, _) in enumerate(samples):
        try:
            image_tf = tf.convert_to_tensor(image_np, dtype=tf.float32)
            heatmap, pred_idx, conf = generate_gradcam(
                model=resnet_grad,
                model_name="ResNet50",
                image_tensor=image_tf,
                preprocess_fn=lambda x: _preprocess_for("resnet50", x),
                last_conv_layer_name=resnet_last_conv,
            )
            heatmap_color, overlay = _create_overlay(image_np, heatmap)
            pred_label = index_to_label.get(pred_idx, str(pred_idx))
            title = f"ResNet50 | Pred: {pred_label} | Conf: {conf:.4f}"
            save_path = gradcam_dir / f"resnet50_gradcam_{idx}.png"
            _save_gradcam_figure(save_path, image_np, heatmap_color, overlay, title)
        except Exception as exc:
            print(f"[GradCAM] Skipping ResNet50 image {idx} due to error: {exc}")
            continue

    print(f"Grad-CAM visualizations saved in: {gradcam_dir}")
    generated_files = sorted(gradcam_dir.glob("*.png"))
    print("Generated Grad-CAM files:")
    for file_path in generated_files:
        print(file_path.resolve())


if __name__ == "__main__":
    main()
