"""Script to upload fine-tuned skin cancer models (.keras and .tflite) to Hugging Face Model Hub."""

import argparse
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    raise ImportError(
        "huggingface_hub package is required to upload models to HF Hub.\n"
        "Please run: pip install huggingface_hub"
    )

import config as cfg

DEFAULT_REPO_ID = getattr(cfg, "HF_REPO_ID", "srushti-projects/skin-cancer-adversarial-defense")

MODEL_FILES_TO_UPLOAD = [
    ("models/mobilenetv2_finetuned.keras", "mobilenetv2_finetuned.keras"),
    ("models/resnet50_finetuned.keras", "resnet50_finetuned.keras"),
    ("outputs/experiments/exp3_proposed_rasc_net/best_model.keras", "best_model.keras"),
    ("models/tflite/mobilenetv2.tflite", "mobilenetv2.tflite"),
    ("models/tflite/resnet50.tflite", "resnet50.tflite"),
    ("models/tflite/rascnet.tflite", "rascnet.tflite"),
]


def upload_models(repo_id: str, token: str = None):
    api = HfApi()
    
    print(f"[HF UPLOADER] Initializing Hugging Face Model Hub repo: '{repo_id}'...")
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"[HF UPLOADER] Repo '{repo_id}' is ready.")
    except Exception as e:
        print(f"[HF UPLOADER] Note on repo creation: {e}")

    backend_dir = cfg.BACKEND_DIR
    success_count = 0

    for rel_path, hf_filename in MODEL_FILES_TO_UPLOAD:
        local_path = backend_dir / rel_path
        if not local_path.exists():
            print(f"[SKIP] Local file not found: {local_path}")
            continue

        print(f"[UPLOADING] {local_path.name} -> {repo_id}/{hf_filename}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_filename,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
            print(f"  [OK] Uploaded {hf_filename} successfully!")
            success_count += 1
        except Exception as err:
            print(f"  [ERROR] Failed to upload {hf_filename}: {err}")

    print(f"\n[SUMMARY] Uploaded {success_count}/{len(MODEL_FILES_TO_UPLOAD)} models to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload trained skin cancer models to Hugging Face Model Hub")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face Model Repo ID (username/repo-name)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Write Token")
    args = parser.parse_args()

    upload_models(repo_id=args.repo_id, token=args.token)
