import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.inference import inspect_bolt, predict_visual, predict_multimodal
from src.confidence import calculate_confidence
from src.decision_engine import decide_inspection
from src.spatial_memory import MEMORY_FILE, save_defect
from src.vision_detector import process_rgb_photo, generate_synthetic_plate_photo

app = FastAPI(
    title="XR Bolt Tightness Inspection API",
    description="Real-time multi-modal AI backend for XR headsets (AR/VR/MR) and industrial digital twins.",
    version="1.0.0"
)

@app.on_event("startup")
def preload_models():
    """Warm up and pre-load ML models, YOLOv8 weights, and OpenCV dependencies at startup."""
    try:
        from src.inference import _get_models
        _get_models()
        from src.vision_detector import get_yolo_model
        get_yolo_model()
        print("XR inspection ML & YOLOv8 models preloaded into RAM successfully.")
    except Exception as e:
        print(f"Model preload warning: {e}")

# Enable CORS for Unity, WebXR, and external browser frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class VisualInspectRequest(BaseModel):
    equipment_id: str = Field(default="HOLED_PLATE_ASSEMBLY", description="ID of parent equipment")
    component_id: str = Field(default="B01", description="Bolt Identifier (e.g. B01, B02, B03, B04)")
    position_3d: List[float] = Field(default=[0.0, 1.0, 0.5], description="Physical [x, y, z] in meters")
    gap_mm: float = Field(default=0.0, ge=0.0, description="Estimated gap in mm from XR depth/camera")
    contact_score: float = Field(default=0.95, ge=0.0, le=1.0, description="Flange contact area score (0 to 1)")
    vision_quality: float = Field(default=0.90, ge=0.0, le=1.0, description="Lighting and camera tracking quality")
    depth_quality: float = Field(default=0.85, ge=0.0, le=1.0, description="Depth sensor point cloud quality")
    registration_quality: float = Field(default=0.95, ge=0.0, le=1.0, description="Spatial anchor registration accuracy")


class TorqueInspectRequest(BaseModel):
    equipment_id: str = Field(default="HOLED_PLATE_ASSEMBLY")
    component_id: str = Field(default="B01")
    position_3d: List[float] = Field(default=[0.0, 1.0, 0.5])
    gap_mm: float = Field(default=0.0, ge=0.0)
    contact_score: float = Field(default=0.95, ge=0.0, le=1.0)
    torque_nm: float = Field(default=28.0, ge=0.0, description="Torque applied in Nm")
    rotation_deg: float = Field(default=0.7, ge=0.0, description="Bolt rotation in degrees during torque test")
    vibration_feature: float = Field(default=0.08, ge=0.0, description="Acoustic/vibration response magnitude")
    temperature_c: float = Field(default=30.0, description="Operating temperature in Celsius")
    vision_quality: float = Field(default=0.90, ge=0.0, le=1.0, description="Lighting and camera tracking quality")
    depth_quality: float = Field(default=0.85, ge=0.0, le=1.0, description="Depth sensor point cloud quality")
    registration_quality: float = Field(default=0.95, ge=0.0, le=1.0, description="Spatial anchor registration accuracy")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "XR Bolt Inspection API",
        "version": "1.0.0"
    }


@app.post("/api/inspect/visual")
def api_inspect_visual(req: VisualInspectRequest):
    """
    Primary XR optical screening endpoint.
    Processes camera/depth sensor features and determines if immediate defect or physical verification is needed.
    """
    try:
        res = inspect_bolt(
            equipment_id=req.equipment_id,
            component_id=req.component_id,
            position_3d=tuple(req.position_3d),
            gap_mm=req.gap_mm,
            contact_score=req.contact_score,
            vision_quality=req.vision_quality,
            depth_quality=req.depth_quality,
            registration_quality=req.registration_quality,
            torque_data=None
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inspect/torque")
def api_inspect_torque(req: TorqueInspectRequest):
    """
    Multi-modal verification endpoint.
    Called when smart wrench or physical tool test is completed. Fuses vision with torque/stiffness response.
    """
    try:
        torque_payload = {
            "torque_nm": req.torque_nm,
            "rotation_deg": req.rotation_deg,
            "vibration_feature": req.vibration_feature,
            "temperature_c": req.temperature_c
        }
        res = inspect_bolt(
            equipment_id=req.equipment_id,
            component_id=req.component_id,
            position_3d=tuple(req.position_3d),
            gap_mm=req.gap_mm,
            contact_score=req.contact_score,
            vision_quality=req.vision_quality,
            depth_quality=req.depth_quality,
            registration_quality=req.registration_quality,
            torque_data=torque_payload
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inspect/photo")
async def api_inspect_photo(file: UploadFile = File(...)):
    """
    Direct RGB Photo Inspection Endpoint:
    Accepts an uploaded photo of the holed plate workpiece, runs computer vision
    feature extraction (gap_mm, contact_score), performs ML inference, updates
    3D spatial memory, and returns an AR-annotated image.
    """
    try:
        contents = await file.read()
        os.makedirs("results", exist_ok=True)
        with open("results/last_uploaded_raw.jpg", "wb") as f_raw:
            f_raw.write(contents)
        res = process_rgb_photo(contents)

        # 3D coordinates map matching CAD holed_plate.obj (B01: TL, B02: TR, B03: BL, B04: BR)
        cad_coords = {
            "B01": (-0.5, 8.0, -4.0),
            "B02": (-0.5, 8.0, 4.0),
            "B03": (-0.5, -8.0, -4.0),
            "B04": (-0.5, -8.0, 4.0)
        }

        # Clear existing defect memory for fresh photo inspection sync
        from src.spatial_memory import clear_defects as clear_all_defects
        clear_all_defects()

        # Synchronize defects to spatial memory
        for b in res.get("bolts", []):
            if "LOOSE" in b["condition"].upper():
                x, y, z = cad_coords.get(b["bolt_id"], (0.0, 0.0, 0.0))
                save_defect(
                    equipment_id="HOLED_PLATE_ASSEMBLY",
                    component_id=b["bolt_id"],
                    x=x,
                    y=y,
                    z=z,
                    defect_type=b["condition"],
                    probability=b["loose_probability"],
                    confidence=b["confidence"],
                    verification="RGB_CAMERA_SCAN"
                )
            else:
                try:
                    clear_single_defect(b["bolt_id"])
                except Exception:
                    pass

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")


@app.post("/api/inspect/sample-photo")
def api_inspect_sample_photo(loose_bolt: str = None, loose_bolts: str = None):
    """Generates a test RGB photo and runs the vision inspection pipeline immediately."""
    try:
        raw = loose_bolts or loose_bolt or "B02,B03"
        bolts_to_loosen = [b.strip() for b in raw.split(",") if b.strip()]
        test_path = generate_synthetic_plate_photo("data/test_plate.jpg", loosen_bolts=bolts_to_loosen)
        res = process_rgb_photo(test_path)

        # 3D coordinates map matching CAD holed_plate.obj (B01: TL, B02: TR, B03: BL, B04: BR)
        cad_coords = {
            "B01": (-0.5, 8.0, -4.0),
            "B02": (-0.5, 8.0, 4.0),
            "B03": (-0.5, -8.0, -4.0),
            "B04": (-0.5, -8.0, 4.0)
        }

        # Clear existing defect memory for fresh sample photo inspection sync
        from src.spatial_memory import clear_defects as clear_all_defects
        clear_all_defects()

        # Synchronize defects to spatial memory
        for b in res.get("bolts", []):
            if "LOOSE" in b["condition"].upper():
                x, y, z = cad_coords.get(b["bolt_id"], (-0.5, 0.0, 0.0))
                save_defect(
                    equipment_id="HOLED_PLATE_ASSEMBLY",
                    component_id=b["bolt_id"],
                    x=x,
                    y=y,
                    z=z,
                    defect_type=b["condition"],
                    probability=b["loose_probability"],
                    confidence=b["confidence"],
                    verification="RGB_CAMERA_SCAN"
                )
            else:
                try:
                    clear_single_defect(b["bolt_id"])
                except Exception:
                    pass

        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inspect/xr-telemetry")
def api_inspect_xr_telemetry(payload: dict = Body(...)):
    """
    Ingests XR Unity / CAD Assembly Telemetry (.json):
    Parses screws, positions, calculates physical gap in mm from X displacement,
    computes contact scores, runs ML model inference, and logs defects to 3D spatial memory.
    """
    try:
        screws = payload.get("screws", [])
        if not screws:
            raise ValueError("No screws found in telemetry JSON")

        bolt_results = []
        defects_count = 0

        # Mapping dictionary
        default_ids = ["B01", "B02", "B03", "B04"]

        for idx, s in enumerate(screws):
            sid = s.get("screw_id", f"Screw_{idx+1:03d}")
            # Map Screw_001 -> B01, etc.
            if "1" in sid:
                bid = "B01"
            elif "2" in sid:
                bid = "B02"
            elif "3" in sid:
                bid = "B03"
            elif "4" in sid:
                bid = "B04"
            else:
                bid = default_ids[idx % len(default_ids)]

            rel_pos = s.get("position_relative_to_assembly", s.get("world_position", {}))
            x_m = float(rel_pos.get("x", 0.0))
            y_m = float(rel_pos.get("y", 0.0))
            z_m = float(rel_pos.get("z", 0.0))

            # Convert meters to millimeters for gap
            gap_mm = round(abs(x_m) * 1000.0, 1)

            if gap_mm >= 6.5:
                contact_score = 0.62
                est_torque = 8.0
                est_rotation = 5.6
            elif gap_mm >= 3.0:
                contact_score = 0.78
                est_torque = 13.0
                est_rotation = 3.4
            else:
                contact_score = 0.96
                est_torque = 28.0
                est_rotation = 0.7

            vision_quality = 0.95
            depth_quality = 0.92
            confidence = calculate_confidence(vision_quality, depth_quality, 0.98)

            ml_res = predict_visual(gap_mm, contact_score)
            cond = ml_res["predicted_condition"]
            is_loose = "LOOSE" in cond.upper()

            if is_loose:
                defects_count += 1
                decision = "CONFIRMED_DEFECT"
                save_defect(
                    equipment_id=payload.get("assembly_name", "HOLED_PLATE_ASSEMBLY"),
                    component_id=bid,
                    x=x_m,
                    y=y_m,
                    z=z_m,
                    defect_type=cond,
                    probability=ml_res["loose_probability"],
                    confidence=confidence,
                    verification="XR_UNITY_CAD_STREAM"
                )
            else:
                decision = "LIKELY_HEALTHY"

            bolt_results.append({
                "bolt_id": bid,
                "screw_id": sid,
                "object_name": s.get("object_name", ""),
                "position_relative_to_assembly": rel_pos,
                "gap_mm": gap_mm,
                "contact_score": contact_score,
                "vision_quality": vision_quality,
                "depth_quality": depth_quality,
                "estimated_torque_nm": est_torque,
                "estimated_rotation_deg": est_rotation,
                "confidence": round(confidence, 2),
                "condition": cond,
                "loose_probability": round(ml_res["loose_probability"], 3),
                "decision": decision
            })

        return {
            "status": "success",
            "source": "XR_UNITY_TELEMETRY",
            "assembly_name": payload.get("assembly_name", "holed_plate (Assembly)"),
            "total_bolts": len(bolt_results),
            "defects_detected": defects_count,
            "bolts": bolt_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process XR telemetry: {str(e)}")


@app.get("/api/inspect/sample-xr-telemetry")
def get_sample_xr_telemetry():
    """Returns the default friend's XR telemetry JSON file."""
    sample_file = os.path.join(os.path.dirname(__file__), "..", "data", "xr_unity_telemetry.json")
    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            data = json.load(f)
        return api_inspect_xr_telemetry(data)
    raise HTTPException(status_code=404, detail="Sample XR telemetry file not found")


@app.get("/api/spatial/defects")
def get_spatial_defects():
    """Returns all recorded defect records with 3D coordinates for XR holographic rendering."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read defect memory: {str(e)}")


@app.delete("/api/spatial/defects")
def clear_spatial_defects():
    """Clears all defect memories (reset button in XR)."""
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump([], f, indent=4)
        return {"status": "success", "message": "Defect memory reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/spatial/defects/{component_id}")
def clear_single_defect(component_id: str):
    """Removes a specific defect record once torqued/resolved."""
    if not os.path.exists(MEMORY_FILE):
        return {"status": "success", "remaining": 0}
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        filtered = [d for d in data if d.get("component_id") != component_id]
        with open(MEMORY_FILE, "w") as f:
            json.dump(filtered, f, indent=4)
        return {"status": "success", "remaining": len(filtered)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
def get_model_info():
    """Returns metadata and test metrics of the trained models."""
    metrics_file = os.path.join(os.path.dirname(__file__), "..", "results", "training_metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {"message": "Metrics file not found"}


# Serve static web files for interactive 3D WebXR Digital Twin
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return {"message": "XR Bolt Inspection API is online. Visit /docs for Swagger UI documentation."}
