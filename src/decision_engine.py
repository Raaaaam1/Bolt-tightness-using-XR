def decide_inspection(
    defect_probability,
    inspection_confidence
):
    """
    Decide whether the inspection result can be accepted
    or whether physical verification is required.
    """

    # Low confidence always requires verification
    if inspection_confidence < 0.80:
        return "PHYSICAL_VERIFICATION"

    # Uncertain defect probability requires verification
    if 0.30 < defect_probability < 0.75:
        return "PHYSICAL_VERIFICATION"

    # High probability of defect
    if defect_probability >= 0.75:
        return "LIKELY_DEFECT"

    # Otherwise
    return "LIKELY_HEALTHY"


# --------------------------------
# TEST CASE
# --------------------------------
if __name__ == "__main__":
    defect_probability = 0.62
    inspection_confidence = 0.74

    decision = decide_inspection(
        defect_probability,
        inspection_confidence
    )

    print("==============================")
    print("XR INSPECTION DECISION")
    print("==============================")

    print(f"Loose probability : {defect_probability:.2f}")
    print(f"Confidence        : {inspection_confidence:.2f}")
    print(f"Decision          : {decision}")

    print("==============================")
