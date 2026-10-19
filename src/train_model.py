import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Paths
DATASET_PATH = "xr_bolt_20000.xlsx"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("XR BOLT TIGHTNESS INSPECTION - MODEL TRAINING PIPELINE")
print("=" * 60)

# 1. Load Dataset
print(f"\n[1/5] Loading dataset from: {DATASET_PATH} ...")
df = pd.read_excel(DATASET_PATH)
print(f"Total samples loaded: {len(df)}")
print(f"Class distribution:\n{df['condition_label'].value_counts().to_dict()}")

# 2. Prepare Features & Target
# Visual Screening Features (available to XR camera / depth sensor)
VISUAL_FEATURES = ["gap_mm_simulated", "contact_score_simulated"]

# Multi-Modal Features (available with torque / wrench verification)
MULTIMODAL_FEATURES = [
    "gap_mm_simulated",
    "contact_score_simulated",
    "torque_nm_simulated",
    "rotation_deg_simulated",
    "vibration_feature_simulated",
    "temperature_c_simulated"
]

target_col = "condition_label"

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df[target_col])
class_names = list(label_encoder.classes_)
print(f"Target classes: {class_names}")

# Train/Test Split (Stratified 80/20)
print("\n[2/5] Performing stratified train/test split (80/20) ...")
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

metrics = {}

# 3. Train Model 1: Visual Screening Model
print("\n[3/5] Training Model 1: Visual Screening Classifier (XR Optical/Depth) ...")
X_train_vis = X_train_df[VISUAL_FEATURES]
X_test_vis = X_test_df[VISUAL_FEATURES]

vis_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
vis_model.fit(X_train_vis, y_train)

y_pred_vis = vis_model.predict(X_test_vis)
vis_acc = accuracy_score(y_test, y_pred_vis)
vis_report = classification_report(y_test, y_pred_vis, target_names=class_names, output_dict=True)

print(f"Visual Model Test Accuracy: {vis_acc * 100:.2f}%")
print("Visual Feature Importances:")
for feat, imp in zip(VISUAL_FEATURES, vis_model.feature_importances_):
    print(f"  - {feat}: {imp:.4f}")

metrics["visual_model"] = {
    "accuracy": float(vis_acc),
    "features": VISUAL_FEATURES,
    "feature_importances": {feat: float(imp) for feat, imp in zip(VISUAL_FEATURES, vis_model.feature_importances_)},
    "report": vis_report
}

# 4. Train Model 2: Multi-Modal Verification Model
print("\n[4/5] Training Model 2: Multi-Modal Verification Classifier (Vision + Torque/Sensors) ...")
X_train_mm = X_train_df[MULTIMODAL_FEATURES]
X_test_mm = X_test_df[MULTIMODAL_FEATURES]

mm_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
mm_model.fit(X_train_mm, y_train)

y_pred_mm = mm_model.predict(X_test_mm)
mm_acc = accuracy_score(y_test, y_pred_mm)
mm_report = classification_report(y_test, y_pred_mm, target_names=class_names, output_dict=True)

print(f"Multi-Modal Model Test Accuracy: {mm_acc * 100:.2f}%")
print("Multi-Modal Feature Importances:")
for feat, imp in zip(MULTIMODAL_FEATURES, mm_model.feature_importances_):
    print(f"  - {feat}: {imp:.4f}")

metrics["multimodal_model"] = {
    "accuracy": float(mm_acc),
    "features": MULTIMODAL_FEATURES,
    "feature_importances": {feat: float(imp) for feat, imp in zip(MULTIMODAL_FEATURES, mm_model.feature_importances_)},
    "report": mm_report
}

# 5. Save Artifacts & Figures
print("\n[5/5] Saving model artifacts, metrics, and plots ...")

joblib.dump(vis_model, os.path.join(MODELS_DIR, "vision_bolt_model.joblib"))
joblib.dump(mm_model, os.path.join(MODELS_DIR, "multimodal_bolt_model.joblib"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.joblib"))

with open(os.path.join(RESULTS_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_vis = confusion_matrix(y_test, y_pred_vis)
im1 = axes[0].imshow(cm_vis, interpolation="nearest", cmap="Blues")
axes[0].set_title(f"Visual Model (Acc: {vis_acc * 100:.1f}%)")
axes[0].set_xticks(range(len(class_names)))
axes[0].set_xticklabels(class_names, rotation=30)
axes[0].set_yticks(range(len(class_names)))
axes[0].set_yticklabels(class_names)
axes[0].set_ylabel("True Condition")
axes[0].set_xlabel("Predicted Condition")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[0].text(j, i, str(cm_vis[i, j]), ha="center", va="center", color="black" if cm_vis[i, j] < cm_vis.max() / 2 else "white")

cm_mm = confusion_matrix(y_test, y_pred_mm)
im2 = axes[1].imshow(cm_mm, interpolation="nearest", cmap="Greens")
axes[1].set_title(f"Multi-Modal Model (Acc: {mm_acc * 100:.1f}%)")
axes[1].set_xticks(range(len(class_names)))
axes[1].set_xticklabels(class_names, rotation=30)
axes[1].set_yticks(range(len(class_names)))
axes[1].set_yticklabels(class_names)
axes[1].set_ylabel("True Condition")
axes[1].set_xlabel("Predicted Condition")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[1].text(j, i, str(cm_mm[i, j]), ha="center", va="center", color="black" if cm_mm[i, j] < cm_mm.max() / 2 else "white")

plt.tight_layout()
cm_plot_path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=300)
plt.close()

print(f"-> Models saved to {MODELS_DIR}/")
print(f"-> Metrics saved to {RESULTS_DIR}/training_metrics.json")
print(f"-> Plot saved to {cm_plot_path}")
print("\nTraining completed successfully!")