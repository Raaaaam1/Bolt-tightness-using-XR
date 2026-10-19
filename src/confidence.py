def calculate_confidence(
    vision_confidence,
    depth_quality,
    registration_quality
):
    """
    Calculate overall inspection confidence.

    All inputs should be between 0 and 1.
    """

    confidence = (
        0.4 * vision_confidence
        + 0.3 * depth_quality
        + 0.3 * registration_quality
    )

    return confidence


# --------------------------------
# TEST EXAMPLE
# --------------------------------
if __name__ == "__main__":
    vision_confidence = 0.62
    depth_quality = 0.70
    registration_quality = 0.95

    confidence = calculate_confidence(
        vision_confidence,
        depth_quality,
        registration_quality
    )

    print("==============================")
    print("XR INSPECTION CONFIDENCE")
    print("==============================")

    print(f"Vision confidence       : {vision_confidence:.2f}")
    print(f"Depth quality           : {depth_quality:.2f}")
    print(f"Registration quality    : {registration_quality:.2f}")
    print(f"Overall confidence      : {confidence:.2f}")

    if confidence < 0.80:
        print("ACTION                  : PHYSICAL VERIFICATION REQUIRED")
    else:
        print("ACTION                  : ACCEPT INSPECTION")

    print("==============================")