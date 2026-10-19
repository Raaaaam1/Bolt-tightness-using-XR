import os
import joblib
import pandas as pd
import numpy as np

from src.confidence import calculate_confidence
from src.decision_engine import decide_inspection
from src.torque_test import analyze_torque_response
from src.fusion import fuse_evidence
from src.spatial_memory import save_defect

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Cached model instances
_vis_model = None
_mm_model = None
_label_encoder = None


def _get_models():
    global _vis_model, _mm_model, _label_encoder
    if _vis_model is None:
        vis_path = os.path.join(MODELS_DIR, "vision_bolt_model.joblib")
        mm_path = os.path.join(MODELS_DIR, "multimodal_bolt_model.joblib")
        enc_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        _vis_model = joblib.load(vis_path)
        _mm_model = joblib.load(mm_path)
        _label_encoder = joblib.load(enc_path)
    return _vis_model, _mm_model, _label_encoder


def predict_visual(gap_mm: float, contact_score: float):
    """Run inference using visual optical/depth features from XR sensor."""
    vis_model, _, encoder = _get_models()
    sample = pd.DataFrame([{
        "gap_mm_simulated": gap_mm,
        "contact_score_simulated": contact_score
    }])
    probs = vis_model.predict_proba(sample)[0]
    classes = encoder.classes_
    class_probs = {cls: float(prob) for cls, prob in zip(classes, probs)}

    # Loose probability is the sum of LOOSE_5MM and LOOSE_7_5MM
    loose_prob = sum(prob for cls, prob in class_probs.items() if "LOOSE" in cls.upper())
    pred_idx = np.argmax(probs)
    predicted_condition = classes[pred_idx]

    return {
        "predicted_condition": predicted_condition,
        "loose_probability": loose_prob,
        "class_probabilities": class_probs
    }


def predict_multimodal(
    gap_mm: float,
    contact_score: float,
    torque_nm: float,
    rotation_deg: float,
    vibration_feature: float,
    temperature_c: float = 30.0
):
    """Run inference combining visual and physical/torque sensory evidence."""
    _, mm_model, encoder = _get_models()
    sample = pd.DataFrame([{
        "gap_mm_simulated": gap_mm,
        "contact_score_simulated": contact_score,
        "torque_nm_simulated": torque_nm,
        "rotation_deg_simulated": rotation_deg,
        "vibration_feature_simulated": vibration_feature,
        "temperature_c_simulated": temperature_c
    }])
    probs = mm_model.predict_proba(sample)[0]
    classes = encoder.classes_
    class_probs = {cls: float(prob) for cls, prob in zip(classes, probs)}

    loose_prob = sum(prob for cls, prob in class_probs.items() if "LOOSE" in cls.upper())
    pred_idx = np.argmax(probs)
    predicted_condition = classes[pred_idx]

    return {
        "predicted_condition": predicted_condition,
        "loose_probability": loose_prob,
        "class_probabilities": class_probs
    }


def inspect_bolt(
    equipment_id: str,
    component_id: str,
    position_3d: tuple,
    gap_mm: float,
    contact_score: float,
    vision_quality: float = 0.90,
    depth_quality: float = 0.85,
    registration_quality: float = 0.95,
    torque_data: dict = None
):
    """
    Full XR inspection pipeline:
    1. Visual screening with ML model
    2. Environmental/registration confidence calculation
    3. Decision engine gating
    4. Optional physical torque verification & fusion
    5. Automatic spatial memory recording if defect confirmed
    """
    # Step 1: Visual ML prediction
    vis_res = predict_visual(gap_mm, contact_score)
    loose_prob = vis_res["loose_probability"]

    # Step 2: Calculate inspection confidence
    confidence = calculate_confidence(
        vision_confidence=vision_quality,
        depth_quality=depth_quality,
        registration_quality=registration_quality
    )

    # Step 3: Decision Engine
    decision = decide_inspection(
        defect_probability=loose_prob,
        inspection_confidence=confidence
    )

    final_verification = "VISUAL_ONLY"
    final_result = vis_res["predicted_condition"]

    # Step 4: Physical verification if torque data is provided
    if torque_data:
        t_nm = float(torque_data.get("torque_nm", 15.0))
        r_deg = float(torque_data.get("rotation_deg", 2.0))
        v_feat = float(torque_data.get("vibration_feature", 0.12))
        temp_c = float(torque_data.get("temperature_c", 30.0))

        # Multimodal prediction
        mm_res = predict_multimodal(gap_mm, contact_score, t_nm, r_deg, v_feat, temp_c)
        final_result = mm_res["predicted_condition"]
        loose_prob = mm_res["loose_probability"]
        final_verification = "TORQUE_FUSION_VERIFIED"

        # Dynamically calculate torque tool measurement confidence
        # Factor in vibration noise and rotation stability
        vibration_penalty = min(0.25, v_feat * 0.6)
        rotation_factor = 1.0 if (0.3 <= r_deg <= 7.5) else 0.85
        torque_factor = 1.0 if t_nm >= 7.0 else (t_nm / 7.0)
        tool_confidence = max(0.50, min(0.99, (0.96 - vibration_penalty) * rotation_factor * torque_factor))

        # Fused overall confidence combining optical tracking & physical torque telemetry
        confidence = round(0.35 * confidence + 0.65 * tool_confidence, 4)

        decision = "CONFIRMED_DEFECT" if loose_prob >= 0.70 else "VERIFIED_HEALTHY"

    # Step 5: Save to spatial memory if defective
    if "LOOSE" in final_result.upper() or decision in ["LIKELY_DEFECT", "CONFIRMED_DEFECT"]:
        x, y, z = position_3d
        save_defect(
            equipment_id=equipment_id,
            component_id=component_id,
            x=x,
            y=y,
            z=z,
            defect_type=final_result,
            probability=loose_prob,
            confidence=confidence,
            verification=final_verification
        )

    return {
        "equipment_id": equipment_id,
        "component_id": component_id,
        "position_3d": position_3d,
        "initial_visual_prediction": vis_res["predicted_condition"],
        "loose_probability": round(loose_prob, 4),
        "confidence": round(confidence, 4),
        "decision": decision,
        "final_result": final_result,
        "verification_mode": final_verification
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING XR INFERENCE PIPELINE")
    print("=" * 60)

    # Case 1: Healthy / Tight Bolt
    res_tight = inspect_bolt(
        equipment_id="PUMP_STATION_A",
        component_id="B01",
        position_3d=(0.5, 1.2, 0.8),
        gap_mm=0.0,
        contact_score=0.98
    )
    print("\nCase 1 (Tight Bolt):")
    for k, v in res_tight.items():
        print(f"  {k}: {v}")

    # Case 2: Clearly Loose Bolt (Visual Only Confirmed)
    res_loose = inspect_bolt(
        equipment_id="PUMP_STATION_A",
        component_id="B02",
        position_3d=(0.6, 1.2, 0.8),
        gap_mm=5.0,
        contact_score=0.75
    )
    print("\nCase 2 (Loose Bolt - Visual):")
    for k, v in res_loose.items():
        print(f"  {k}: {v}")

    # Case 3: Borderline / Low Confidence -> Triggers Physical Verification
    res_verify = inspect_bolt(
        equipment_id="PUMP_STATION_A",
        component_id="B03",
        position_3d=(0.7, 1.2, 0.8),
        gap_mm=2.5,
        contact_score=0.85,
        vision_quality=0.60,  # Lower quality triggers physical verification
        torque_data={"torque_nm": 9.2, "rotation_deg": 5.1, "vibration_feature": 0.25}
    )
    print("\nCase 3 (Physical Verification & Fusion):")
    for k, v in res_verify.items():
        print(f"  {k}: {v}")
