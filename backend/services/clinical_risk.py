"""Rule-Based Clinical Risk Assessment Service.

Computes a transparent, rule-based clinical risk score based on patient metadata and CNN predictions.
Note: This is a decision-support heuristic score, NOT a neural network prediction.
"""

from typing import Any, Dict, List


def calculate_clinical_risk_score(metadata: Dict[str, Any], cnn_predicted_code: str) -> Dict[str, Any]:
    """Calculate transparent clinical risk score and generate recommendations based on HAM10000 metadata."""
    score = 0
    explanations: List[str] = []

    # 1. Approximate Age Rule (age_approx)
    try:
        age = int(metadata.get("age_approx", 0))
    except (ValueError, TypeError):
        age = 0

    if age > 60:
        score += 2
        explanations.append(f"Age > 60 ({age} yrs) (+2 pts)")
    elif 40 <= age <= 60:
        score += 1
        explanations.append(f"Age 40–60 ({age} yrs) (+1 pt)")

    # 2. Anatomical Sites Hierarchy (anatom_site_1, anatom_site_2, anatom_site_3)
    sites = [
        str(metadata.get("anatom_site_1", "")).lower(),
        str(metadata.get("anatom_site_2", "")).lower(),
        str(metadata.get("anatom_site_3", "")).lower(),
    ]
    high_risk_sites = ["head/neck", "face", "scalp", "ear"]
    if any(s in site for s in high_risk_sites for site in sites if site):
        score += 1
        explanations.append("High-exposure anatomical site (Head/Neck/Face) (+1 pt)")

    # 3. Melanocytic Status Rule (melanocytic: True / False)
    melanocytic = metadata.get("melanocytic")
    if melanocytic is True or str(melanocytic).lower() in ["true", "yes", "1"]:
        score += 2
        explanations.append("Melanocytic / Pigmented lesion (+2 pts)")

    # 4. Concomitant Biopsy Rule (concomitant_biopsy: True / False)
    biopsy = metadata.get("concomitant_biopsy")
    if biopsy is True or str(biopsy).lower() in ["true", "yes", "1"]:
        score += 2
        explanations.append("Concomitant biopsy indicated (+2 pts)")

    # 5. Clinical Diagnoses Hierarchy (diagnosis_1 to diagnosis_5)
    diagnoses = [
        str(metadata.get("diagnosis_1", "")).lower(),
        str(metadata.get("diagnosis_2", "")).lower(),
        str(metadata.get("diagnosis_3", "")).lower(),
        str(metadata.get("diagnosis_4", "")).lower(),
        str(metadata.get("diagnosis_5", "")).lower(),
    ]

    suspicious_keywords = ["melanoma", "basal cell carcinoma", "bcc", "actinic keratosis", "akiec"]
    for diag in diagnoses:
        if diag and any(kw in diag for kw in suspicious_keywords):
            score += 2
            explanations.append(f"Suspicious clinical diagnosis history ({diag.title()}) (+2 pts)")
            break

    # 6. CNN Image Prediction Rule (RASC-Net Proposed)
    predicted_code_lower = str(cnn_predicted_code).lower()
    if predicted_code_lower in ["mel", "akiec", "bcc"]:
        score += 2
        explanations.append(f"RASC-Net High-Risk Visual Classification ({cnn_predicted_code.upper()}) (+2 pts)")

    # Risk Categorization
    if score <= 2:
        risk_level = "LOW RISK"
        recommendation = "Routine dermatological evaluation recommended for standard skin monitoring."
    elif 3 <= score <= 5:
        risk_level = "MODERATE RISK"
        recommendation = "Dermoscopic examination and close clinical follow-up advised within 2–4 weeks."
    else:
        risk_level = "HIGH RISK"
        recommendation = "PROMPT URGENT DERMATOLOGICAL EVALUATION and histological biopsy strongly recommended."

    return {
        "score": score,
        "risk_level": risk_level,
        "explanation": explanations,
        "recommendation": recommendation,
        "patient_metadata": metadata,
    }
