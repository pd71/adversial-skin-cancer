"""Rule-Based Clinical Risk Assessment Service.

Computes a transparent, rule-based clinical risk score based on patient metadata and CNN predictions.
Note: This is a decision-support heuristic score, NOT a neural network prediction.
"""

from typing import Any, Dict, List


def calculate_clinical_risk_score(metadata: Dict[str, Any], cnn_predicted_code: str) -> Dict[str, Any]:
    """Calculate transparent clinical risk score and generate recommendations."""
    score = 0
    explanations: List[str] = []

    # 1. Age Rule
    try:
        age = int(metadata.get("age", 0))
    except (ValueError, TypeError):
        age = 0

    if age > 60:
        score += 2
        explanations.append("Age > 60 (+2 pts)")
    elif 40 <= age <= 60:
        score += 1
        explanations.append("Age 40–60 (+1 pt)")

    # 2. Family History Rule
    fam_hist = metadata.get("family_history")
    if fam_hist is True or str(fam_hist).lower() in ["true", "yes"]:
        score += 2
        explanations.append("Family history of skin cancer (+2 pts)")

    # 3. Previous Skin Cancer Diagnosis Rule
    prev_cancer = metadata.get("previous_cancer")
    if prev_cancer is True or str(prev_cancer).lower() in ["true", "yes"]:
        score += 3
        explanations.append("Previous skin cancer diagnosis (+3 pts)")

    # 4. Symptoms Rule
    symptoms = metadata.get("symptoms", [])
    if isinstance(symptoms, str):
        try:
            import json
            symptoms = json.loads(symptoms)
        except Exception:
            symptoms = [s.strip() for s in symptoms.split(",")]

    symptom_weights = {
        "Bleeding": (2, "Symptom: Bleeding (+2 pts)"),
        "Ulceration": (2, "Symptom: Ulceration (+2 pts)"),
        "Rapid Growth": (2, "Symptom: Rapid Growth (+2 pts)"),
        "Change in Color": (1, "Symptom: Change in Color (+1 pt)"),
        "Change in Size": (1, "Symptom: Change in Size (+1 pt)"),
        "Pain": (1, "Symptom: Pain (+1 pt)"),
        "Itching": (1, "Symptom: Itching (+1 pt)"),
    }

    if isinstance(symptoms, list):
        for sym in symptoms:
            if sym in symptom_weights:
                pts, msg = symptom_weights[sym]
                score += pts
                explanations.append(msg)

    # 5. Lesion Duration Rule
    duration = str(metadata.get("duration", ""))
    if any(d in duration for d in ["6–12 months", "6-12 months", "More than 1 year", "> 1 year"]):
        score += 1
        explanations.append("Lesion duration > 6 months (+1 pt)")

    # 6. CNN Image Prediction Rule
    predicted_code_lower = str(cnn_predicted_code).lower()
    if predicted_code_lower in ["mel", "akiec", "bcc"]:
        score += 2
        explanations.append(f"CNN High-Risk Prediction ({cnn_predicted_code.upper()}) (+2 pts)")

    # Risk Categories
    if score <= 2:
        risk_level = "LOW RISK"
        recommendation = "Routine dermatology consultation recommended for standard evaluation."
    elif 3 <= score <= 5:
        risk_level = "MODERATE RISK"
        recommendation = "Dermoscopic examination and close clinical monitoring advised within 2–4 weeks."
    else:
        risk_level = "HIGH RISK"
        recommendation = "PROMPT URGENT DERMATOLOGY EVALUATION and histological biopsy strongly recommended."

    return {
        "score": score,
        "risk_level": risk_level,
        "explanation": explanations,
        "recommendation": recommendation,
        "patient_metadata": metadata,
    }
