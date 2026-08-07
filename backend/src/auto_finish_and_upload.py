import argparse
import os
import time
from pathlib import Path
import config as cfg
from upload_to_huggingface import upload_models

def main(repo_id: str = None, token: str = None):
    if repo_id is None:
        repo_id = getattr(cfg, "HF_REPO_ID", "holypreet/rasc-net")
    if token is None:
        token = os.environ.get("HF_TOKEN")

    exp3_best = cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras"
    print(f"[AUTO WATCHER] Monitoring training progress... Will upload to HF '{repo_id}' upon completion.")
    
    while True:
        if exp3_best.exists():
            time.sleep(30)  # brief pause after write
            print(f"\n[AUTO WATCHER] Training finished! Uploading updated models to Hugging Face Model Hub ({repo_id})...")
            upload_models(repo_id=repo_id, token=token)
            print(f"\n[AUTO WATCHER] ALL DONE! High-accuracy models uploaded to https://huggingface.co/{repo_id}")
            break
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    main(repo_id=args.repo_id, token=args.token)

