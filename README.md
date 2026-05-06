# Skin Cancer Classification Research Pipeline (HAM10000)

This project is a clean, reproducible skeleton for an end-to-end deep learning research workflow using the HAM10000 dataset.

Planned pipeline stages:
- Train lightweight CNN baselines: `MobileNetV2`, `EfficientNetB0`, `ResNet50`
- Evaluate and compare models
- Apply adversarial attacks: `FGSM`, `PGD`, `CW`
- Add defenses: feature squeezing + adversarial training
- Generate explainability outputs with Grad-CAM

## Project Structure

```text
SkinCancerProject/
|-- data/
|   |-- HAM10000_images_part_1/
|   |-- HAM10000_images_part_2/
|   `-- HAM10000_metadata.csv
|-- src/
|   |-- config.py
|   |-- utils.py
|   |-- data_loader.py
|   |-- train_models.py
|   |-- evaluate_models.py
|   |-- attacks.py
|   |-- defenses.py
|   |-- gradcam.py
|   `-- run_pipeline.py
|-- models/
|-- outputs/
|   |-- plots/
|   |-- metrics/
|   `-- adversarial_examples/
|-- requirements.txt
|-- README.md
`-- .gitignore
```

## Environment Setup (Windows - PowerShell)

Run all commands from project root (`IPD_Final_Project`):

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

3. Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

4. Install required packages:

```powershell
pip install -r requirements.txt
```

5. Verify TensorFlow installation:

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

6. Run project environment check script:

```powershell
python src/check_env.py
```

If any package check fails, run:

```powershell
pip install -r requirements.txt
```

### Linux/macOS (optional reference)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python src/check_env.py
```

## Dataset Placement

Keep dataset files exactly here (already expected by config):

- `data/HAM10000_images_part_1/`
- `data/HAM10000_images_part_2/`
- `data/HAM10000_metadata.csv`

The class label column in metadata is expected to be `dx`.

## Run Commands (Current Skeleton)

From project root:

```bash
python src/run_pipeline.py
```

This currently:
- sets random seeds for reproducibility,
- validates dataset paths,
- ensures `models/` and `outputs/` subfolders exist.

## Upcoming Module Responsibilities

- `src/data_loader.py`: metadata loading, path mapping, train/val/test split
- `src/train_models.py`: training for MobileNetV2, EfficientNetB0, ResNet50
- `src/evaluate_models.py`: metrics, confusion matrix, comparison plots
- `src/attacks.py`: FGSM, PGD, CW attack evaluation
- `src/defenses.py`: feature squeezing and adversarial training hooks
- `src/gradcam.py`: Grad-CAM visual explanation generation

## Notes

- All paths are relative to project root and should work on Windows/Linux/macOS.
- `.gitignore` excludes local dataset, trained models, and generated outputs by default.
