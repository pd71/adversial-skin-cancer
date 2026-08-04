# 📡 REST API Documentation

The RASC-Net backend API is implemented using Flask and provides endpoints for skin cancer classification, clinical risk evaluation, adversarial attack simulation, defense filtering, and scientific benchmark metrics.

---

## 📍 Base URL
`http://localhost:5000`

---

## 1. Health Check Endpoint

### `GET /api/health`
Verifies server status and loaded model configurations.

#### Request Example
```http
GET /api/health HTTP/1.1
Host: localhost:5000
```

#### Response Example (`200 OK`)
```json
{
  "status": "healthy",
  "models_loaded": {
    "mobilenetv2": true,
    "resnet50": true,
    "rasc_net_proposed": true
  },
  "timestamp": "2026-08-01T16:48:00Z"
}
```

---

## 2. Skin Lesion Prediction & Clinical Assessment Endpoint

### `POST /api/predict`
Classifies a skin lesion image using a specified CNN model and computes patient clinical risk based on submitted patient metadata.

#### Request Headers
`Content-Type: multipart/form-data`

#### Form Data Fields
* `image`: File (JPEG/PNG skin lesion image)
* `model`: String (`"RASC-Net Proposed"`, `"MobileNetV2"`, `"ResNet50"`, `"Soft Voting Ensemble"`, `"RASC-Net Baseline"`, `"RASC-Net Regularized"`)
* `patient_age`: Integer (e.g., `55`)
* `patient_sex`: String (`"male"`, `"female"`)
* `anatomical_site`: String (`"torso"`, `"face"`, `"upper extremity"`, `"lower extremity"`, `"head/neck"`)
* `lesion_diameter`: Float (e.g., `7.5`)
* `evolution_changed`: Boolean (`true` / `false`)
* `personal_history_cancer`: Boolean (`true` / `false`)

#### Response Example (`200 OK`)
```json
{
  "status": "success",
  "selected_model": "RASC-Net Proposed",
  "prediction": "mel",
  "lesion_name": "Melanoma",
  "confidence": 88.42,
  "confidence_level": "HIGH",
  "top_3_predictions": [
    { "class": "mel", "lesion_name": "Melanoma", "confidence": 88.42 },
    { "class": "nv", "lesion_name": "Melanocytic nevi", "confidence": 8.15 },
    { "class": "bkl", "lesion_name": "Benign keratosis-like lesions", "confidence": 3.43 }
  ],
  "clinical_assessment": {
    "score": 7,
    "risk_level": "HIGH RISK",
    "recommendation": "Urgent dermatological referral and excisional biopsy strongly advised.",
    "risk_factors": [
      "Melanocytic / pigmented lesion suspicious for Melanoma",
      "Lesion diameter > 6mm (ABCDE rule criteria)",
      "Reported recent evolution/changes in size, shape, or color",
      "Personal or family history of skin malignancy"
    ]
  }
}
```

---

## 3. Adversarial Attack Simulation Endpoint

### `POST /api/attack`
Simulates FGSM, PGD, or CW gradient perturbations on an uploaded image.

#### Request Headers
`Content-Type: multipart/form-data`

#### Form Data Fields
* `image`: File
* `attack_type`: String (`"FGSM"`, `"PGD"`, `"CW"`)
* `epsilon`: Float (`0.01` – `0.05`)

#### Response Example (`200 OK`)
```json
{
  "status": "success",
  "attack_type": "FGSM",
  "epsilon": 0.03,
  "clean_prediction": "Melanoma (88.4%)",
  "adversarial_prediction": "Melanocytic nevi (92.1%)",
  "attack_successful": true,
  "adversarial_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACt..."
}
```

---

## 4. Defense Filtering Pipeline Endpoint

### `POST /api/defense`
Applies bit-depth reduction, Gaussian blur, and JPEG compression filtering to sanitize an adversarial image.

#### Request Headers
`Content-Type: multipart/form-data`

#### Form Data Fields
* `image`: File

#### Response Example (`200 OK`)
```json
{
  "status": "success",
  "original_prediction": "Melanocytic nevi (92.1%)",
  "defended_prediction": "Melanoma (84.6%)",
  "recovery_successful": true,
  "defended_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACt..."
}
```

---

## 5. Scientific Benchmark Metrics Endpoint

### `GET /api/metrics`
Returns evaluation benchmarks, confidence intervals, McNemar test results, and plot file URIs across all 6 models.

#### Response Example (`200 OK`)
```json
{
  "status": "success",
  "models_benchmark": [
    {
      "model_name": "RASC-Net Proposed (Exp 3)",
      "clean_accuracy": 65.57,
      "ci_95": "[62.77%, 68.46%]",
      "fgsm_accuracy": 48.00,
      "pgd_accuracy": 6.00,
      "cw_accuracy": 60.00,
      "robustness_score": 38.00,
      "recovery_rate": 79.69,
      "ece_score": 0.1768,
      "parameters": 2875307,
      "model_size_mb": 33.25
    }
  ]
}
```
