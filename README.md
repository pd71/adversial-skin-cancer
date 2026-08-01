# RASC-Net: Residual Attention Skin Cancer Network & Adversarial Defense Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework and Clinical Decision Support System (CDSS) for skin lesion classification under adversarial threats. Incorporates **RASC-Net** (Residual Attention Skin Cancer Network), **Curriculum Adversarial Training**, multi-stage image defense, Explainable AI (Grad-CAM), and hospital-grade PDF clinical reporting.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training & Ablation Experiments](#training--ablation-experiments)
- [Scientific Benchmark & Evaluation](#scientific-benchmark--evaluation)
- [Launching Web Application](#launching-web-application)
- [Key Findings & Results](#key-findings--results)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## 🔬 Overview

Skin lesion classification using Deep Convolutional Neural Networks (CNNs) achieves high diagnostic accuracy but is vulnerable to imperceptible adversarial perturbations. **RASC-Net** addresses this challenge by combining:
1. **Lightweight Residual Attention Backbone**: Channel and Spatial Attention modules ($2.88\text{M}$ parameters).
2. **Curriculum Adversarial Training**: Percentage-based adaptive FGSM schedule ($\epsilon=0.01$).
3. **Data Regularization**: MixUp augmentation ($\alpha=0.2$) and Label Smoothing ($\epsilon=0.1$).
4. **Pre-Processing Defense Pipeline**: 4-bit Depth Reduction + Gaussian Blur + JPEG Compression (70% quality).
5. **Clinical Decision Support System**: Integration of patient metadata (age, sex, anatomical site, diameter, evolution, history) for rule-based risk scoring and hospital-grade PDF report export.

---

## 📁 Repository Structure

```
IPD_Final_Project/
├── backend/
│   ├── models/                    # Pretrained & fine-tuned model checkpoints (.keras)
│   ├── outputs/
│   │   ├── experiments/           # Experiment 1, 2, 3 outputs (checkpoints, CSVs, manifests)
│   │   ├── evaluation/            # Step 7 evaluation JSONs, figures, master table
│   │   ├── metrics/               # Evaluation JSONs & text reports
│   │   └── plots/                 # High-resolution publication plots
│   ├── routes/
│   │   ├── predict.py             # Predict & Clinical Risk API
│   │   ├── gradcam.py             # Grad-CAM Heatmap Generation API
│   │   ├── attack.py              # Adversarial Attack Simulation API
│   │   ├── defense.py             # Image Defense Filtering API
│   │   └── metrics.py             # Scientific Evaluation Dashboard API
│   ├── services/
│   │   └── clinical_risk.py       # Rule-based clinical risk scoring engine
│   └── src/
│       ├── config.py              # Global project hyperparameters & paths
│       ├── data_loader.py         # HAM10000 dataset loading & preprocessing
│       ├── robust_skin_net.py     # RASC-Net architecture (Custom Layers)
│       ├── run_attacks.py         # FGSM, PGD, CW attack implementations
│       ├── train_models.py        # Model training routines & callbacks
│       ├── run_ablation_study.py  # Ablation study execution script
│       └── evaluate_all_models.py # Master 6-model benchmark script
├── frontend/                      # React + Vite + Tailwind CSS Web Application
│   ├── src/
│   │   ├── components/            # UI Components & Hospital Report Modal
│   │   ├── pages/                 # Predict, Attack, Defense, Metrics pages
│   │   └── services/              # Axios API service handlers
│   └── package.json
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### System Requirements
* **OS**: Windows 10/11, Linux, or macOS
* **Python**: 3.10+
* **Node.js**: 18.x+ (for frontend)

### 1. Python Environment
```bash
# Clone Repository
git clone https://github.com/user/skin-cancer-adversarial-defense.git
cd skin-cancer-adversarial-defense

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate # Windows

# Install Backend Dependencies
pip install tensorflow==2.15.0 numpy pandas scikit-learn matplotlib seaborn scipy pillow flask flask-cors reportlab
```

### 2. Frontend Environment
```bash
cd frontend
npm install
cd ..
```

---

## 📊 Dataset Preparation

1. Download the **HAM10000** dataset from Kaggle or ISIC Archive.
2. Structure the dataset directory as follows under `backend/data/`:
```
backend/data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

---

## 🚀 Training & Ablation Experiments

To run the full RASC-Net Ablation Study (Exp 1: Baseline, Exp 2: Regularized, Exp 3: Proposed):

```bash
python backend/src/run_ablation_study.py
```

Outputs will be automatically saved under `backend/outputs/experiments/`.

---

## 🔬 Scientific Benchmark & Evaluation

To reproduce all benchmarks, 95% bootstrap confidence intervals, McNemar significance tests, and 9 publication-quality figures across all 6 model configurations:

```bash
python backend/src/evaluate_all_models.py
```

Results are saved to `backend/outputs/evaluation/` and `backend/outputs/plots/unified_benchmark/`.

---

## 💻 Launching Web Application

### Launch Backend Server
```bash
python backend/app.py
# Backend runs on http://localhost:5000
```

### Launch Frontend Client
```bash
cd frontend
npm run dev
# Frontend runs on http://localhost:5173
```

---

## 🏆 Key Findings & Results

| Model Architecture | Clean Acc (95% CI) | FGSM Acc | PGD Acc | CW Acc | Recovery Rate | ECE | Params | Latency | Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2** | 69.06% [66.26%, 71.96%] | 32.00% | 0.00% | 10.00% | 21.59% | 0.0242 | 2.42M | 223ms | 22.7MB |
| **ResNet50** | 75.85% [73.25%, 78.44%] | 12.00% | 0.00% | 10.00% | 75.18% | 0.0263 | 23.85M | 404ms | 203.9MB |
| **Soft Voting Ensemble** | 69.26% [66.46%, 72.16%] | 28.00% | 0.00% | 10.00% | 77.55% | 0.1315 | 26.27M | 32.6ms | 226.6MB |
| **RASC-Net Baseline (Exp 1)** | 49.30% [46.50%, 52.40%] | 2.00% | 0.00% | 20.00% | 71.88% | 0.0922 | 2.88M | 141ms | 33.3MB |
| **RASC-Net Regularized (Exp 2)**| 58.98% [56.09%, 62.08%] | 8.00% | 0.00% | 36.00% | 66.69% | 0.0701 | 2.88M | 124ms | 33.3MB |
| **RASC-Net Proposed (Exp 3)** | **65.57% [62.77%, 68.46%]** | **48.00%** | **6.00%** | **60.00%** | **79.69%** | 0.1768 | **2.88M** | **138ms** | **33.3MB** |

---

## 📜 Citation & Acknowledgments

If you find this codebase or research useful in your work, please cite:

```bibtex
@article{rascnet2026,
  title={Effects of Adversarial Attacks on Skin Cancer Classification using Deep Learning: A Robust Attention Approach},
  author={Antigravity AI Team},
  journal={IEEE Transactions on Medical Imaging (Submitted)},
  year={2026}
}
```
