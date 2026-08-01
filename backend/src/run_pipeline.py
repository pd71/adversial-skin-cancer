"""Pipeline entry point (skeleton).

This script sets up reproducibility, validates paths, and provides
placeholders for the next implementation stages:
1) train models
2) evaluate models
3) run attacks/defenses
4) generate Grad-CAM outputs
"""

import random

import numpy as np
import tensorflow as tf

from config import (
    DATA_DIR,
    FORCE_RETRAIN,
    METADATA_PATH,
    MODELS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    dataset_paths_exist,
    ensure_project_dirs,
)
from train_models import train_efficientnetb0, train_mobilenetv2, train_resnet50, train_rasc_net



def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Set reproducible random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main() -> None:
    """Run the high-level research pipeline."""
    set_global_seed(RANDOM_SEED)
    ensure_project_dirs()

    print("=== Skin Cancer Research Pipeline ===")
    print(f"Data directory: {DATA_DIR}")
    print(f"Metadata path : {METADATA_PATH}")
    print(f"Models dir    : {MODELS_DIR}")
    print(f"Outputs dir   : {OUTPUTS_DIR}")
    print(f"Force retrain : {FORCE_RETRAIN}")

    if not dataset_paths_exist():
        print("\n[ERROR] Dataset paths are missing.")
        print("Please ensure the following exist under project root:")
        print("- data/HAM10000_images_part_1/")
        print("- data/HAM10000_images_part_2/")
        print("- data/HAM10000_metadata.csv")
        return

    print("\n[OK] Dataset paths found.")
    mobilenet_final_model = MODELS_DIR / "mobilenetv2_finetuned.keras"
    if mobilenet_final_model.exists() and not FORCE_RETRAIN:
        print("MobileNetV2 already trained, skipping...")
    else:
        print("[RUN] Training MobileNetV2...")
        mobilenet_artifacts = train_mobilenetv2()
        print("\n[DONE] MobileNetV2 training complete.")
        print(f"Best checkpoint : {mobilenet_artifacts['best_checkpoint_path']}")
        print(f"Final model     : {mobilenet_artifacts['final_model_path']}")
        print(f"Training plot   : {mobilenet_artifacts['training_plot_path']}")
        print(f"Metrics report  : {mobilenet_artifacts['report_path']}")
        print(f"Confusion plot  : {mobilenet_artifacts['confusion_plot_path']}")

    efficientnet_final_model = MODELS_DIR / "efficientnetb0_finetuned.keras"
    if efficientnet_final_model.exists() and not FORCE_RETRAIN:
        print("EfficientNetB0 already trained, skipping...")
    else:
        print("\n[RUN] Training EfficientNetB0...")
        efficientnet_artifacts = train_efficientnetb0()
        print("\n[DONE] EfficientNetB0 training complete.")
        print(f"Best checkpoint : {efficientnet_artifacts['best_checkpoint_path']}")
        print(f"Final model     : {efficientnet_artifacts['final_model_path']}")
        print(f"Training plot   : {efficientnet_artifacts['training_plot_path']}")
        print(f"Metrics report  : {efficientnet_artifacts['report_path']}")
        print(f"Confusion plot  : {efficientnet_artifacts['confusion_plot_path']}")

    resnet_final_model = MODELS_DIR / "resnet50_finetuned.keras"
    if resnet_final_model.exists() and not FORCE_RETRAIN:
        print("ResNet50 already trained, skipping...")
    else:
        print("\n[RUN] Training ResNet50...")
        resnet_artifacts = train_resnet50()
        print("\n[DONE] ResNet50 training complete.")
        print(f"Best checkpoint : {resnet_artifacts['best_checkpoint_path']}")
        print(f"Final model     : {resnet_artifacts['final_model_path']}")
        print(f"Training plot   : {resnet_artifacts['training_plot_path']}")
        print(f"Metrics report  : {resnet_artifacts['report_path']}")
        print(f"Confusion plot  : {resnet_artifacts['confusion_plot_path']}")

    rasc_final_model = MODELS_DIR / "rasc_net_finetuned.keras"
    if rasc_final_model.exists() and not FORCE_RETRAIN:
        print("\nRASC-Net already trained, skipping...")
    else:
        print("\n[RUN] Training RASC-Net (Custom Architecture)...")
        rasc_artifacts = train_rasc_net()
        print("\n[DONE] RASC-Net training complete.")
        print(f"Best checkpoint : {rasc_artifacts['best_checkpoint_path']}")
        print(f"Final model     : {rasc_artifacts['final_model_path']}")
        print(f"Training plot   : {rasc_artifacts['training_plot_path']}")
        print(f"Metrics report  : {rasc_artifacts['report_path']}")
        print(f"Confusion plot  : {rasc_artifacts['confusion_plot_path']}")



if __name__ == "__main__":
    main()
