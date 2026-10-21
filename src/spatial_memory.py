import json
from datetime import datetime


MEMORY_FILE = "results/defect_memory.json"


def save_defect(
    equipment_id,
    component_id,
    x,
    y,
    z,
    defect_type,
    probability,
    confidence,
    verification
):
    """
    Store a confirmed or suspected defect
    together with its physical 3D location.
    """

    # Create the results folder/file if needed
    try:
        with open(MEMORY_FILE, "r") as file:
            memory = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        memory = []

    defect = {
        "equipment_id": equipment_id,
        "component_id": component_id,

        "position": {
            "x": x,
            "y": y,
            "z": z
        },

        "defect_type": defect_type,
        "probability": probability,
        "confidence": confidence,
        "verification": verification,

        "timestamp": datetime.now().isoformat()
    }

    # Avoid duplicate rows for the same component_id
    memory = [d for d in memory if d.get("component_id") != component_id]
    memory.append(defect)

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

    print("Defect stored successfully.")


# --------------------------------
# TEST CASE
# --------------------------------
if __name__ == "__main__":
    save_defect(
        equipment_id="Machine_01",
        component_id="Bolt_B2",

        x=1.25,
        y=0.84,
        z=2.10,

        defect_type="LOOSE_BOLT",

        probability=0.85,
        confidence=0.91,

        verification="TORQUE_TEST"
    )