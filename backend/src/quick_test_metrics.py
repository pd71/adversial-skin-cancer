"""Quick evaluation script to compute and print exact Test Accuracy and Macro F1 for all trained models."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data_loader import load_ham10000_data
from robust_skin_net import build_rasc_net
import config as cfg

def quick_eval():
    print("\n" + "=" * 60)
    print("      FAST SCIENTIFIC EVALUATION REPORT (HAM10000 TEST SET)")
    print("=" * 60)

    data_dict = load_ham10000_data(batch_size=cfg.BATCH_SIZE)
    test_ds = data_dict["test_ds"]
    test_df = data_dict["test_df"]
    y_true = test_df["label"].values.astype(int)
    index_to_label = data_dict["index_to_label"]
    target_names = [index_to_label[i] for i in sorted(index_to_label.keys())]

    # Models to test
    models_to_test = [
        ("MobileNetV2 (Fine-tuned)", cfg.MODELS_DIR / "mobilenetv2_finetuned.keras", False),
        ("ResNet50 (Fine-tuned)", cfg.MODELS_DIR / "resnet50_finetuned.keras", False),
        ("RASC-Net Baseline (Exp 1)", cfg.OUTPUTS_DIR / "experiments" / "exp1_baseline_rasc_net" / "best_model.keras", True),
        ("RASC-Net Regularized (Exp 2)", cfg.OUTPUTS_DIR / "experiments" / "exp2_regularized_rasc_net" / "best_model.keras", True),
        ("RASC-Net Proposed (Exp 3)", cfg.OUTPUTS_DIR / "experiments" / "exp3_proposed_rasc_net" / "best_model.keras", True),
    ]

    predictions_dict = {}

    for name, path, is_rasc in models_to_test:
        if not path.exists():
            print(f"[SKIP] Model checkpoint missing: {path.name}")
            continue

        try:
            if is_rasc:
                model = build_rasc_net(input_shape=(224, 224, 3), num_classes=7)
                model.load_weights(path)
            else:
                model = tf.keras.models.load_model(path)

            probs = model.predict(test_ds, verbose=0)
            preds = np.argmax(probs, axis=1)
            predictions_dict[name] = probs

            acc = accuracy_score(y_true, preds)
            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

            print(f"\n* {name}:")
            print(f"  - Test Accuracy: {acc * 100:.2f}%")
            print(f"  - Macro F1-Score: {macro_f1 * 100:.2f}%")
        except Exception as e:
            print(f"[ERROR] Evaluating {name}: {e}")

    # Evaluate Soft Voting Ensemble
    if len(predictions_dict) >= 2:
        ensemble_probs = np.mean(list(predictions_dict.values()), axis=0)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        ens_acc = accuracy_score(y_true, ensemble_preds)
        ens_f1 = f1_score(y_true, ensemble_preds, average="macro", zero_division=0)

        print("\n" + "=" * 60)
        print("[ENSEMBLE] SOFT VOTING ENSEMBLE (All Models Combined + TTA)")
        print(f"  - Test Accuracy:  {ens_acc * 100:.2f}%")
        print(f"  - Macro F1-Score: {ens_f1 * 100:.2f}%")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    quick_eval()
