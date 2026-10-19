def fuse_evidence(
    vision_probability,
    torque_probability
):
    """
    Combine visual and physical inspection evidence.
    """

    vision_weight = 0.30
    torque_weight = 0.70

    final_probability = (
        vision_weight * vision_probability
        + torque_weight * torque_probability
    )

    return final_probability


# --------------------------------
# TEST CASE
# --------------------------------
if __name__ == "__main__":
    vision_probability = 0.62
    torque_probability = 0.95

    final_probability = fuse_evidence(
        vision_probability,
        torque_probability
    )

    print("==============================")
    print("EVIDENCE FUSION")
    print("==============================")

    print(f"Vision probability : {vision_probability:.2f}")
    print(f"Torque probability : {torque_probability:.2f}")
    print(f"Final probability  : {final_probability:.2f}")

    if final_probability >= 0.75:
        print("FINAL RESULT       : LOOSE BOLT CONFIRMED")
    else:
        print("FINAL RESULT       : NO CONFIRMED DEFECT")

    print("==============================")