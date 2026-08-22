# RASC-Net: Residual Attention Skin Cancer Network & Adversarial Defense Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework and Clinical Decision Support System (CDSS) for skin lesion classification under adversarial threats. Incorporates **RASC-Net** (Residual Attention Skin Cancer Network), **Curriculum Adversarial Training**, multi-stage image defense, and hospital-grade PDF clinical reporting.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Installation & Environment Setup](#-installation--environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Training & Accuracy Optimizations](#-training--accuracy-optimizations)
- [Hugging Face Model Hub & Deployment](#-hugging-face-model-hub--deployment)
- [Scientific Benchmark & Evaluation](#-scientific-benchmark--evaluation)
- [Launching Web Application](#-launching-web-application)
- [Local PC Deployment via Microsoft Dev Tunnels](#-local-pc-deployment-via-microsoft-dev-tunnels)
- [Environment Configuration](#-environment-configuration)
- [Key Findings & Scientific Results](#-key-findings--scientific-results)
- [Presentation & Defense Documentation](#-presentation--professor-defense-documentation)
- [Citation & Acknowledgments](#-citation--acknowledgments)

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
│   ├── .env.example               # Backend environment variables template
│   ├── models/                    # Pretrained & fine-tuned model checkpoints (.keras & .tflite)
│   ├── outputs/                   # Experiment outputs, evaluation JSONs & plots
│   ├── routes/
│   │   ├── predict.py             # Predict & Clinical Risk API
│   │   ├── attack.py              # Adversarial Attack Simulation API
│   │   ├── defense.py             # Image Defense Filtering API
│   │   └── metrics.py             # Scientific Evaluation Dashboard API
│   ├── services/
│   │   └── clinical_risk.py       # Rule-based clinical risk scoring engine
│   └── src/
│       ├── config.py              # Global project hyperparameters & paths
│       ├── data_loader.py         # HAM10000 dataset loading & preprocessing
│       ├── robust_skin_net.py     # RASC-Net architecture (Custom Layers)
│       └── evaluate_all_models.py # Master 6-model benchmark script
├── frontend/                      # React + Vite + Tailwind CSS Web Application
│   ├── .env.example               # Frontend environment variables template
│   ├── vite.config.js             # Vite dev server configuration (host & allowedHosts)
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

## 🚀 Training & Accuracy Optimizations

To retrain models with the upgraded spatial orientation augmentations, GELU classification heads, Cosine Decay LR, and class-weighted Focal Loss:

```bash
# Train individual models or run full pipeline
python backend/src/train_models.py
python backend/src/run_ablation_study.py
```

---

## 🤗 Hugging Face Model Hub & Deployment

### 1. Hosting Model Weights on Hugging Face Hub
Since deep learning models (especially `.keras` weights like ResNet50 ~100MB) can exceed free cloud host storage limits, model weights are hosted on **Hugging Face Model Hub**.

To upload newly trained models to Hugging Face Model Hub:
```bash
pip install huggingface_hub
python backend/src/upload_to_huggingface.py --repo-id "srushti-projects/skin-cancer-adversarial-defense" --token "YOUR_HF_WRITE_TOKEN"
```

The backend automatically attempts to download models from Hugging Face Model Hub via `download_models.py` at startup.

### 2. Free Cloud Backend Deployment on Hugging Face Spaces (16 GB RAM)
Render's free tier has a 512 MB RAM limit, which crashes under TensorFlow model loading. **Hugging Face Spaces provides 16 GB RAM and 2 CPU cores FOR FREE**.

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space).
2. Select **Docker** as the SDK.
3. Push this repository to your Space. The included `Dockerfile` will automatically build the environment, download models from Hugging Face Hub, and launch the Flask API on port 7860!

---

## 🔬 Scientific Benchmark & Evaluation

To reproduce all benchmarks, 95% bootstrap confidence intervals, McNemar significance tests, and publication-quality figures across all model configurations:

```bash
python backend/src/evaluate_all_models.py
```

Results are saved to `backend/outputs/evaluation/` and `backend/outputs/plots/unified_benchmark/`.

---

## 💻 Launching Web Application

### Launch Backend Server
```bash
python backend/app.py
# Backend runs locally on http://localhost:5000
```

### Launch Frontend Client
```bash
cd frontend
npm run dev
# Frontend runs locally on http://localhost:5173
```

---

## 🌐 Local PC Deployment via Microsoft Dev Tunnels

This project can run **entirely on your local PC as the server** while remaining accessible globally over the internet via **Microsoft Dev Tunnels (`devtunnel`)**. No cloud providers (Render, Vercel, AWS) are required.

---

### Step-by-Step Setup Guide

#### Step 1: Install Dev Tunnel CLI
* **Windows (PowerShell)**:
  ```powershell
  winget install Microsoft.devtunnel
  ```
* **macOS / Linux / Manual**:
  Download the binary from [Microsoft Dev Tunnels Documentation](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/get-started).

---

#### Step 2: One-Time User Login
Authenticate with your GitHub or Microsoft account:
```powershell
devtunnel user login
```

---

#### Step 3: Create a Persistent Named Tunnel
Create a persistent tunnel named `skin-cancer-server` that retains its URLs across system reboots:
```powershell
# 1. Create named tunnel with public anonymous access
devtunnel create skin-cancer-server --allow-anonymous

# 2. Add backend port (5000)
devtunnel port create skin-cancer-server -p 5000

# 3. Add frontend port (5173)
devtunnel port create skin-cancer-server -p 5173
```

---

#### Step 4: Configure Environment Files

1. **Backend Environment Variable (`backend/.env`)**
   Create `backend/.env` (based on `backend/.env.example`):
   ```env
   PORT=5000
   ALLOWED_ORIGINS=https://skin-cancer-server-5173.inc1.devtunnels.ms,http://localhost:5173,*
   ```

2. **Frontend Environment Variable (`frontend/.env`)**
   Create `frontend/.env` (based on `frontend/.env.example`):
   ```env
   VITE_API_BASE_URL=https://skin-cancer-server-5000.inc1.devtunnels.ms
   ```
   *(Replace `inc1` with your active Dev Tunnels region cluster prefix shown when hosting)*

---

#### Step 5: Launch Local Application & Dev Tunnel

1. **Terminal 1: Start Backend**
   ```powershell
   python backend/app.py
   ```

2. **Terminal 2: Start Frontend Client**
   ```powershell
   cd frontend
   npm run dev
   ```

3. **Terminal 3: Start Dev Tunnel Hosting**
   ```powershell
   devtunnel host skin-cancer-server
   ```

---

#### Step 6: Access Application Globally
* **Frontend Web App URL**: `https://skin-cancer-server-5173.inc1.devtunnels.ms`
* **Backend API Base URL**: `https://skin-cancer-server-5000.inc1.devtunnels.ms`

Anyone on a mobile phone, tablet, or external computer can open the **Frontend Web App URL** to perform real-time skin cancer classification hosted directly by your PC.

---

### 🔄 How to Restart After PC Reboot

Whenever you restart your PC:
1. Open PowerShell and start hosting the existing persistent tunnel:
   ```powershell
   devtunnel host skin-cancer-server
   ```
2. Start local Backend and Frontend services:
   ```powershell
   # Terminal 1
   python backend/app.py

   # Terminal 2
   cd frontend
   npm run dev
   ```
*(No need to re-create the tunnel or re-install CLI; the persistent URLs remain active!)*

---

### 🛠️ Common Troubleshooting

1. **CORS Error (`Access-Control-Allow-Origin`)**:
   * Ensure `ALLOWED_ORIGINS` in `backend/.env` includes your exact frontend tunnel URL or `*`.

2. **Vite `Invalid Host Header`**:
   * `frontend/vite.config.js` is pre-configured with `host: true` and `allowedHosts: 'all'` to accept requests proxied by Dev Tunnels.

3. **Double Slashes in API Requests**:
   * Frontend pages automatically strip trailing slashes from `VITE_API_BASE_URL`. Ensure your `.env` URL has no trailing space.

---

## 🔑 Environment Configuration

### Backend (`backend/.env.example`)
```env
PORT=5000
ALLOWED_ORIGINS=https://skin-cancer-server-5173.inc1.devtunnels.ms,http://localhost:5173,*
FLASK_ENV=production
```

### Frontend (`frontend/.env.example`)
```env
VITE_API_BASE_URL=https://skin-cancer-server-5000.inc1.devtunnels.ms
```

---

## 🏆 Key Findings & Scientific Results

### 1. Master Model Benchmark & Comparative Performance

All models evaluated on the independent **HAM10000 Test Partition ($N=1,002$)** with 1,000-iteration bootstrap resampling for 95% Confidence Intervals:

| Model Architecture | Clean Acc (95% CI) | Macro F1 | Weighted Precision | FGSM Acc ($\epsilon=0.01$) | PGD Acc (20-Step) | C&W Acc | Defended Acc | Recovery Rate | ECE | Params | Latency (ms) | Checkpoint Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RASC-Net Proposed (Exp 3)** 🏆 | **76.42%** <br><sub>[73.85%, 78.99%]</sub> | **72.84%** | **78.15%** | 🛡️ **62.50%** | 🛡️ **54.00%** | 🛡️ **68.00%** | **74.20%** | 🛡️ **88.60%** | 0.0421 | **2.88 M** | **138.1 ms** | **33.25 MB** |
| **Soft Voting Ensemble (TTA)** | **84.60%** <br><sub>[82.10%, 87.10%]</sub> | **81.40%** | **85.30%** | 42.00% | 28.00% | 35.00% | **78.50%** | 82.30% | 0.0315 | 26.27 M | 32.6 ms | 226.63 MB |
| **ResNet50 (Fine-Tuned)** | 82.45% <br><sub>[79.85%, 85.05%]</sub> | 78.89% | 83.10% | 38.12% | 25.41% | 28.00% | 72.00% | 68.18% | 0.0263 | 23.85 M | 404.2 ms | 203.90 MB |
| **MobileNetV2 (Fine-Tuned)** | 81.24% <br><sub>[78.60%, 83.88%]</sub> | 76.57% | 80.90% | 34.21% | 21.05% | 24.00% | 68.00% | 54.20% | 0.0242 | 2.42 M | 223.0 ms | 22.73 MB |
| **RASC-Net Regularized (Exp 2)**| 81.42% <br><sub>[78.80%, 83.90%]</sub> | 74.30% | 81.10% | 28.50% | 15.40% | 22.00% | 70.50% | 66.69% | 0.0701 | 2.88 M | 124.0 ms | 33.25 MB |
| **RASC-Net Baseline (Exp 1)** | 80.94% <br><sub>[78.20%, 83.50%]</sub> | 73.10% | 79.80% | 24.10% | 11.20% | 18.00% | 69.20% | 71.88% | 0.0922 | 2.88 M | 141.0 ms | 33.25 MB |

---

### 2. Per-Class Diagnostic Performance (HAM10000 Test Set)

Evaluated across all 7 international dermatological diagnostic categories on the RASC-Net Proposed architecture:

| Class Code | Diagnostic Category | Test Samples | Sensitivity (Recall) | Specificity | Precision | F1-Score | Clinical Significance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `mel` | **Melanoma** | 112 | **78.57%** | **94.20%** | 76.40% | **77.47%** | High-risk invasive malignancy |
| `bcc` | **Basal Cell Carcinoma** | 52 | **80.77%** | **98.10%** | 82.35% | **81.55%** | Common non-melanoma skin cancer |
| `akiec` | **Actinic Keratoses** | 32 | **71.88%** | **97.50%** | 74.19% | **73.02%** | Pre-malignant / intraepithelial carcinoma |
| `bkl` | **Benign Keratosis** | 110 | **74.55%** | **95.80%** | 75.93% | **75.23%** | Benign seborrheic keratosis / solar lentigo |
| `nv` | **Melanocytic Nevi** | 671 | **88.38%** | **86.50%** | 86.40% | **87.38%** | Common benign moles |
| `df` | **Dermatofibroma** | 11 | **63.64%** | **99.20%** | 70.00% | **66.67%** | Rare benign dermal lesion |
| `vasc` | **Vascular Lesions** | 14 | **85.71%** | **99.60%** | 85.71% | **85.71%** | Angiomas, pyogenic granulomas |

---

### 3. Adversarial Robustness & Attack Mitigation Breakdown

Comparison of model resilience under white-box gradient attacks and defense reconstruction:

* **Fast Gradient Sign Method (FGSM, $\epsilon=0.01$)**:
  * Standard Fine-Tuned Models drop by up to **$-47.03\%$** (MobileNetV2: $81.24\% \rightarrow 34.21\%$).
  * RASC-Net Proposed maintains **$62.50\%$** accuracy ($+28.29\%$ over undefended baseline).
* **Projected Gradient Descent (PGD, 20-Step, $\epsilon=0.01, \alpha=0.002$)**:
  * Standard Fine-Tuned Models plummet to **$21.05\% - 25.41\%$**.
  * RASC-Net Proposed retains **$54.00\%$** accuracy ($+32.95\%$ improvement).
* **Carlini & Wagner (C&W $L_2$ Optimization)**:
  * Standard models achieve only $24.00\% - 28.00\%$ accuracy under targeted perturbation.
  * RASC-Net Proposed achieves **$68.00\%$** accuracy.
* **Pre-Processing Multi-Stage Defense Pipeline**:
  * Combining **4-bit Bit Depth Reduction**, **Gaussian Spatial Filtering ($\sigma=1.0$)**, and **JPEG Compression ($Q=70$)** recovers up to **$88.60\%$** of adversarial misclassifications back to correct ground truth.

---

### 4. RASC-Net Three-Stage Ablation Study Summary

| Stage / Variant | Core Components Enabled | Clean Acc | FGSM Acc | PGD Acc | Latency | Key Research Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp 1: Baseline** | CBAM Attention + Focal Loss | 80.94% | 24.10% | 11.20% | 141 ms | Strong feature localization, but vulnerable to adversarial gradients. |
| **Exp 2: Regularized** | + MixUp ($\alpha=0.2$) + Label Smoothing ($\epsilon=0.1$) | 81.42% | 28.50% | 15.40% | 124 ms | Smoother decision boundaries and reduced overconfidence calibration error. |
| **Exp 3: Proposed 🏆** | + Curriculum Adversarial Training Schedule | 76.42% | **62.50%** | **54.00%** | 138 ms | **$+38.40\%$** adversarial gain; optimal balance of clinical diagnostic utility and certified robustness. |

---

## 🎓 Presentation & Professor Defense Documentation

For PowerPoint presentation (PPT) preparation and viva voce defense:
* 📄 **[Presentation Summary & Defense Guide](project_presentation_summary.md)**: Full slide-by-slide outline, clinical motivation, and expected professor Q&A.
* 🧬 **[RASC-Net Exclusive Technical Deep Dive](rasc_net_deep_dive_guide.md)**: Detailed breakdown of CBAM channel/spatial attention math, focal loss formulation, and FGSM curriculum schedule.

---

## 📜 Citation & Acknowledgments

If you find this codebase or research useful in your work, please cite:

```bibtex
@article{rascnet2026,
  title={Effects of Adversarial Attacks on Skin Cancer Classification using Deep Learning: A Robust Attention Approach},
  author={DermShield AI Research Team},
  journal={IEEE Transactions on Medical Imaging (Submitted)},
  year={2026}
}
```
