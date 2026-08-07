import os
import requests

# Base directory points to backend/ folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HF_REPO = "holypreet/rasc-net"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"


MODELS = {
    # Original TensorFlow Keras Models (Hugging Face Model Hub primary, GitHub fallback)
    "models/mobilenetv2_finetuned.keras": [
        f"{HF_BASE_URL}/mobilenetv2_finetuned.keras",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/mobilenetv2_finetuned.keras",
    ],
    "models/resnet50_finetuned.keras": [
        f"{HF_BASE_URL}/resnet50_finetuned.keras",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/resnet50_finetuned.keras",
    ],
    "outputs/experiments/exp3_proposed_rasc_net/best_model.keras": [
        f"{HF_BASE_URL}/best_model.keras",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/best_model.keras",
    ],
    # TensorFlow Lite Models (Primary Inference Engine)
    "models/tflite/mobilenetv2.tflite": [
        f"{HF_BASE_URL}/mobilenetv2.tflite",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/mobilenetv2.tflite",
    ],
    "models/tflite/resnet50.tflite": [
        f"{HF_BASE_URL}/resnet50.tflite",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/resnet50.tflite",
    ],
    "models/tflite/rascnet.tflite": [
        f"{HF_BASE_URL}/rascnet.tflite",
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/rascnet.tflite",
    ],
}



def download_file(urls, relative_path):
    full_path = os.path.join(BASE_DIR, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        print(f"Model already exists, skipping download: {relative_path}")
        return

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        print(f"Attempting download: {relative_path} from {url}...")
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(full_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded successfully: {relative_path}")
            return
        except Exception as e:
            print(f"Failed attempt for {relative_path} ({url}): {e}")
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except Exception:
                    pass

    print(f"[WARNING] Unable to download {relative_path} from any source.")


def download_models():
    # Ensure target directories exist
    os.makedirs(os.path.join(BASE_DIR, "models", "tflite"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "outputs", "experiments", "exp3_proposed_rasc_net"), exist_ok=True)

    for relative_path, urls in MODELS.items():
        full_path = os.path.join(BASE_DIR, relative_path)
        if os.path.exists(full_path):
            print(f"Model already exists, skipping download: {relative_path}")
            continue

        download_file(urls, relative_path)


    # Check and log TFLite status at startup (ASCII safe)
    mobilenet_tflite = os.path.exists(os.path.join(BASE_DIR, "models", "tflite", "mobilenetv2.tflite"))
    resnet_tflite = os.path.exists(os.path.join(BASE_DIR, "models", "tflite", "resnet50.tflite"))
    rascnet_tflite = os.path.exists(os.path.join(BASE_DIR, "models", "tflite", "rascnet.tflite"))

    print("\nFound TFLite models:")
    print(f"  [{'OK' if mobilenet_tflite else 'MISSING'}] MobileNet")
    print(f"  [{'OK' if resnet_tflite else 'MISSING'}] ResNet50")
    print(f"  [{'OK' if rascnet_tflite else 'MISSING'}] RASC-Net\n")


if __name__ == "__main__":
    download_models()
