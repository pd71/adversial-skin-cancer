import os
import requests

# Base directory (project root: parent directory of backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = {
    "backend/models/mobilenetv2_finetuned.keras":
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/mobilenetv2_finetuned.keras",

    "backend/models/resnet50_finetuned.keras":
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/resnet50_finetuned.keras",

    "backend/outputs/experiments/exp3_proposed_rasc_net/best_model.keras":
        "https://github.com/srushti-projects/skin-cancer-adversarial-defense/releases/download/v1.0/best_model.keras"
}


def download_file(url, relative_path):
    full_path = os.path.join(BASE_DIR, relative_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        print(f"Already exists: {relative_path}")
        return

    print(f"Downloading {relative_path}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(full_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Successfully downloaded: {relative_path}")


def download_models():
    for relative_path, url in MODELS.items():
        download_file(url, relative_path)


if __name__ == "__main__":
    download_models()
