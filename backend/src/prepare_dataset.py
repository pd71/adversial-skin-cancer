"""Script to locate downloaded HAM10000 dataset from kagglehub and structure it under backend/data/."""

import os
import shutil
from pathlib import Path
import config as cfg

def prepare_dataset():
    cfg.ensure_project_dirs()
    target_data_dir = cfg.DATA_DIR
    target_data_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dataset_paths_exist():
        print(f"[OK] HAM10000 dataset already configured at {target_data_dir}")
        return True

    print("[DATASET PREPARATION] Checking kagglehub cache...")
    try:
        import kagglehub
        kaggle_path = Path(kagglehub.dataset_download('kmader/skin-cancer-mnist-ham10000'))
        print(f"[DATASET PREPARATION] Found kagglehub download at: {kaggle_path}")

        # Move / copy metadata and image directories
        metadata_src = kaggle_path / "HAM10000_metadata.csv"
        part1_src = kaggle_path / "HAM10000_images_part_1"
        part2_src = kaggle_path / "HAM10000_images_part_2"

        if not part1_src.exists() and (kaggle_path / "ham10000_images_part_1").exists():
            part1_src = kaggle_path / "ham10000_images_part_1"
        if not part2_src.exists() and (kaggle_path / "ham10000_images_part_2").exists():
            part2_src = kaggle_path / "ham10000_images_part_2"

        if metadata_src.exists():
            shutil.copy2(metadata_src, cfg.METADATA_PATH)
            print(f"  [COPIED] {cfg.METADATA_PATH}")

        if part1_src.exists() and not cfg.IMAGES_PART_1_DIR.exists():
            try:
                os.symlink(part1_src, cfg.IMAGES_PART_1_DIR, target_is_directory=True)
                print(f"  [LINKED] {cfg.IMAGES_PART_1_DIR} -> {part1_src}")
            except Exception:
                shutil.copytree(part1_src, cfg.IMAGES_PART_1_DIR)
                print(f"  [COPIED] {cfg.IMAGES_PART_1_DIR}")

        if part2_src.exists() and not cfg.IMAGES_PART_2_DIR.exists():
            try:
                os.symlink(part2_src, cfg.IMAGES_PART_2_DIR, target_is_directory=True)
                print(f"  [LINKED] {cfg.IMAGES_PART_2_DIR} -> {part2_src}")
            except Exception:
                shutil.copytree(part2_src, cfg.IMAGES_PART_2_DIR)
                print(f"  [COPIED] {cfg.IMAGES_PART_2_DIR}")

        if cfg.dataset_paths_exist():
            print("[SUCCESS] HAM10000 dataset is fully prepared for training!")
            return True
        else:
            print("[WARNING] Dataset structure incomplete after copy.")
            return False

    except Exception as e:
        print(f"[ERROR] Failed dataset preparation: {e}")
        return False

if __name__ == "__main__":
    prepare_dataset()
