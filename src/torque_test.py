def analyze_torque_response(torque_nm, rotation_deg):
    """
    Analyse the response of a bolt during
    physical torque verification.
    """

    # Avoid division by zero
    if rotation_deg <= 0:
        rotation_deg = 0.01

    stiffness = torque_nm / rotation_deg

    # Temporary experimental threshold
    # We will determine this from real measurements later.
    if stiffness < 0.25:
        probability = 0.95
    elif stiffness < 0.50:
        probability = 0.70
    else:
        probability = 0.10

    return stiffness, probability


# --------------------------------
# TEST CASE
# --------------------------------
if __name__ == "__main__":
    torque = 1.15
    rotation = 7.2

    stiffness, probability = analyze_torque_response(
        torque,
        rotation
    )

    print("==============================")
    print("TORQUE VERIFICATION")
    print("==============================")

    print(f"Applied torque : {torque:.2f} Nm")
    print(f"Rotation       : {rotation:.2f} degrees")
    print(f"Response ratio : {stiffness:.3f}")
    print(f"Loose probability: {probability:.2f}")

    print("==============================")