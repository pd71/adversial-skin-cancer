"""Central project configuration for the Skin Cancer pipeline."""

from pathlib import Path


# -----------------------------
# Reproducibility
# -----------------------------
RANDOM_SEED = 42


# -----------------------------
# Paths (relative to project root)
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_PART_1_DIR = DATA_DIR / "HAM10000_images_part_1"
IMAGES_PART_2_DIR = DATA_DIR / "HAM10000_images_part_2"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"

BACKEND_DIR = PROJECT_ROOT / "backend"
SRC_DIR = BACKEND_DIR / "src"
MODELS_DIR = BACKEND_DIR / "models"
OUTPUTS_DIR = BACKEND_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"
ADVERSARIAL_EXAMPLES_DIR = OUTPUTS_DIR / "adversarial_examples"


# -----------------------------
# Dataset / labels
# -----------------------------
TARGET_COLUMN = "dx"
IMAGE_FILENAME_COLUMN = "image_id"

# HAM10000 classes (canonical short labels)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)


# -----------------------------
# Training defaults (skeleton)
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

MODEL_CANDIDATES = ["MobileNetV2", "EfficientNetB0", "ResNet50", "RASC-Net"]

FORCE_RETRAIN = False
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1

# -----------------------------
# MixUp Data Augmentation & Label Smoothing
# -----------------------------
ENABLE_MIXUP = True
MIXUP_ALPHA = 0.2
ENABLE_LABEL_SMOOTHING = True
LABEL_SMOOTHING = 0.1




# -----------------------------
# Adversarial Training Curriculum
# -----------------------------
ENABLE_ADVERSARIAL_TRAINING = True
FGSM_EPSILON = 0.01
ADVERSARIAL_CURRICULUM = [
    (0.25, 0.00),   # First 25% epochs -> 100% Clean
    (0.50, 0.25),   # Next 25% epochs -> 75% Clean / 25% FGSM
    (1.00, 0.50),   # Final 50% epochs -> 50% Clean / 50% FGSM
]

# -----------------------------
# Test-Time Augmentation (TTA) & Evaluation
# -----------------------------
ENABLE_TTA = True
TTA_STEPS = 4

# -----------------------------
# Hugging Face Hub Integration
# -----------------------------
HF_REPO_ID = "srushti-projects/skin-cancer-adversarial-defense"
HF_HUB_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main"

PGD_EPSILON = 0.01
PGD_ALPHA = 0.002
PGD_STEPS = 10
CW_CONFIDENCE = 0.0
CW_STEPS = 100




def ensure_project_dirs() -> None:
    """Create writable output/model directories if missing."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ADVERSARIAL_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def dataset_paths_exist() -> bool:
    """Check whether expected dataset files/folders are present."""
    required_paths = [IMAGES_PART_1_DIR, IMAGES_PART_2_DIR, METADATA_PATH]
    return all(path.exists() for path in required_paths)
