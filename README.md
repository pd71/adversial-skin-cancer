# RASC-Net: Residual Attention Skin Cancer Network & Adversarial Defense Framework

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework and Clinical Decision Support System (CDSS) for skin lesion classification under adversarial threats. Incorporates **RASC-Net** (Residual Attention Skin Cancer Network), **Curriculum Adversarial Training**, multi-stage image defense, and hospital-grade PDF clinical reporting.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation & Environment Setup](#installation--environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training & Accuracy Optimizations](#training--accuracy-optimizations)
- [Hugging Face Model Hub & Deployment](#-hugging-face-model-hub--deployment)
- [Scientific Benchmark & Evaluation](#scientific-benchmark--evaluation)
- [Launching Web Application](#launching-web-application)
- [Local PC Deployment via Microsoft Dev Tunnels](#-local-pc-deployment-via-microsoft-dev-tunnels)
- [Environment Configuration](#-environment-configuration)
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

## 🏆 Key Findings & Results

| Model Architecture | Clean Acc (95% CI) | FGSM Acc | PGD Acc | CW Acc | Recovery Rate | ECE | Params | Latency | Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
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
