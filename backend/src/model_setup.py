from pathlib import Path

from config import MODELS_DIR


REQUIRED_MODEL_FILES = [
    MODELS_DIR / "mobilenetv2_finetuned.keras",
    MODELS_DIR / "mobilenetv2_class_mapping.json",
    MODELS_DIR / "resnet50_finetuned.keras",
    MODELS_DIR / "resnet50_class_mapping.json",
]


def missing_model_files() -> list[Path]:
    """Return the list of required model files that are missing."""
    return [path for path in REQUIRED_MODEL_FILES if not path.exists()]


def print_model_status() -> int:
    """Print a brief model file status report and return exit code."""
    missing = missing_model_files()
    if not missing:
        print("All required backend model files are present in backend/models/.")
        return 0

    print("Missing required model files in backend/models/:")
    for path in missing:
        print(f"  - {path.name}")

    print("\nThese files are intentionally excluded from git."
          " Copy them into backend/models/ before running the backend.")
    print("If you do not have the files, generate them with the model training pipeline:")
    print("  python backend/src/run_pipeline.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(print_model_status())
