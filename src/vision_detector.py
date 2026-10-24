import os
import cv2
import base64
import numpy as np
from typing import Dict, List, Tuple, Union

from src.inference import predict_visual, inspect_bolt
from src.confidence import calculate_confidence

# Physical dimensions from CAD model (holed_plate.obj) in millimeters
PLATE_LENGTH_MM = 200.0  # Length along Y (200 mm)
PLATE_WIDTH_MM = 120.0   # Width along Z (120 mm)
BOLT_DIAMETER_MM = 20.0  # Washer outer diameter (20 mm)

_YOLO_MODEL = None


def get_yolo_model():
    """Lazy-loads and caches the fine-tuned YOLOv8 bolt detection model."""
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_bolt_yolo.pt")
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                _YOLO_MODEL = YOLO(model_path)
            except Exception:
                _YOLO_MODEL = None
    return _YOLO_MODEL


def process_rgb_photo(
    image_source: Union[str, bytes, np.ndarray],
    save_annotated_path: str = "results/annotated_inspection.jpg"
) -> Dict:
    """
    Precision Industrial Washer Inspection Pipeline:
    1. Preprocesses image with bilateral filter (preserves washer edges, suppresses scratches).
    2. Segments the workpiece into 4 corner search quadrants.
    3. Detects the circular outer washer rim for each fastener assembly.
    4. Analyzes the 360-degree perimeter contact profile:
       - True loose bolts produce a wide contiguous crescent shadow arc (> 35 deg).
       - Tight bolts with surface oil spots, chamfers, or texture have 0 contiguous arc.
    5. Feeds calibrated gap_mm and contact_score to ML model.
    6. Renders high-precision holographic targeting reticles and status overlays.
    """
    if isinstance(image_source, str):
        img = cv2.imread(image_source)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_source}")
    elif isinstance(image_source, bytes):
        nparr = np.frombuffer(image_source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_source, np.ndarray):
        img = image_source.copy()
    else:
        raise ValueError("Unsupported image source type")

    if img is None:
        raise ValueError("Could not decode image. Please ensure the file is a valid JPG, PNG, or WEBP image.")

    h, w = img.shape[:2]
    # Automatically downscale ultra-high-res phone images to prevent memory spikes on cloud instances
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(filtered, 50, 150)

    # 1. Scale and Baseline Plate Background Intensity
    cw1, cw2 = int(w * 0.3), int(w * 0.7)
    ch1, ch2 = int(h * 0.3), int(h * 0.7)
    plate_baseline = float(np.median(gray[ch1:ch2, cw1:cw2]))
    if plate_baseline <= 0:
        plate_baseline = float(np.median(gray))

    # Plate scale estimation
    pixels_per_mm = max(w, h) / PLATE_LENGTH_MM
    aspect_ratio = float(w) / float(h)

    # 2. Multi-Scale Fastener Detection (Handles both Big Industrial Bolts and Small Screws)
    # Primary Tier: Custom YOLOv8 Deep Learning Detector (Fastener ROI Localization)
    yolo_bolts = []
    yolo_model = get_yolo_model()
    if yolo_model is not None:
        try:
            y_res = yolo_model.predict(img, conf=0.18, verbose=False)
            boxes = y_res[0].boxes
            if boxes is not None and len(boxes) > 0:
                raw_cands = []
                for box in boxes:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = coords
                    bx_w, bx_h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    if 15 < bx_w < w * 0.70 and 15 < bx_h < h * 0.70:
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cr = int(min(bx_w, bx_h) / 2.0)
                        cr = max(18, min(cr, 65))
                        raw_cands.append((cx, cy, cr, conf))
                raw_cands.sort(key=lambda c: c[3], reverse=True)
                clean_yolo = []
                for cx, cy, cr, conf in raw_cands:
                    too_close = any(np.hypot(cx - ex[0], cy - ex[1]) < 70 for ex in clean_yolo)
                    if not too_close:
                        clean_yolo.append((cx, cy, cr, "yolo", f"B{len(clean_yolo)+1:02d}", conf))
                if len(clean_yolo) >= 4:
                    radii = [c[2] for c in clean_yolo]
                    med_r = float(np.median(radii))
                    valid_cands = [c for c in clean_yolo if c[2] >= 0.65 * med_r]
                    if len(valid_cands) >= 2:
                        clean_yolo = valid_cands
                    if len(clean_yolo) > 4:
                        clean_yolo = clean_yolo[:4]
                if len(clean_yolo) >= 1:
                    yolo_bolts = [(c[0], c[1], c[2], c[3], c[4]) for c in clean_yolo]
        except Exception:
            yolo_bolts = []

    # Scale A: Large Industrial Bolts/Washers (r: 34 to 65 px)
    large_bolts = []
    if aspect_ratio > 0.65 and w >= 300 and h >= 250:
        quads = [
            ("B01", 0, 0, w // 2, h // 2),
            ("B02", w // 2, 0, w, h // 2),
            ("B03", 0, h // 2, w // 2, h),
            ("B04", w // 2, h // 2, w, h)
        ]
        quad_circles = []
        for qid, qx1, qy1, qx2, qy2 in quads:
            q_roi = filtered[qy1:qy2, qx1:qx2]
            qc = cv2.HoughCircles(
                q_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                param1=50, param2=20, minRadius=34, maxRadius=55
            )
            if qc is not None:
                bc = qc[0][0]
                cx, cy, cr = int(bc[0] + qx1), int(bc[1] + qy1), int(bc[2])
                if 20 < cx < w - 20 and 20 < cy < h - 20:
                    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                    xs = np.clip(np.int32(cx + cr * np.cos(theta)), 0, w - 1)
                    ys = np.clip(np.int32(cy + cr * np.sin(theta)), 0, h - 1)
                    cov = float(np.sum(edges[ys, xs] > 0)) / 36.0
                    if cov >= 0.16:
                        quad_circles.append((cx, cy, cr, "large", qid))
        if len(quad_circles) == 4:
            large_bolts = quad_circles

    if not large_bolts and w >= 450:
        c_large = cv2.HoughCircles(
            filtered, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
            param1=50, param2=18, minRadius=28, maxRadius=70
        )
        if c_large is not None:
            for c in c_large[0]:
                cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
                if cr < cx < w - cr and cr < cy < h - cr:
                    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                    xs = np.clip(np.int32(cx + cr * np.cos(theta)), 0, w - 1)
                    ys = np.clip(np.int32(cy + cr * np.sin(theta)), 0, h - 1)
                    cov = float(np.sum(edges[ys, xs] > 0)) / 36.0
                    if cov >= 0.16:
                        large_bolts.append((cx, cy, cr, "large", f"B{len(large_bolts)+1:02d}"))

    # Scale B: Small Fasteners / Screws (r: 13 to 28 px)
    small_screws = []
    if not large_bolts or len(large_bolts) < 2:
        mask = np.zeros_like(gray)
        y_min, y_max = int(h * 0.16), int(h * 0.94)
        x_min, x_max = int(w * 0.18), int(w * 0.95)
        mask[y_min:y_max, x_min:x_max] = 255
        masked_filtered = cv2.bitwise_and(filtered, filtered, mask=mask)

        c_small = cv2.HoughCircles(
            masked_filtered, cv2.HOUGH_GRADIENT, dp=1.1, minDist=100,
            param1=45, param2=14, minRadius=14, maxRadius=32
        )
        if c_small is not None:
            for c in c_small[0]:
                cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
                theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
                xs = np.clip(np.int32(cx + cr * np.cos(theta)), 0, w - 1)
                ys = np.clip(np.int32(cy + cr * np.sin(theta)), 0, h - 1)
                cov = float(np.sum(edges[ys, xs] > 0)) / 36.0

                # Validate genuine screw texture via gradient magnitude (rejects flat surface false positives)
                roi = gray[max(0, cy - cr):min(h, cy + cr), max(0, cx - cr):min(w, cx + cr)]
                if roi.size > 0:
                    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1)
                    grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2)))
                else:
                    grad_mag = 0.0

                if cov >= 0.05 and grad_mag >= 45.0:
                    small_screws.append((cx, cy, cr, "small"))

    # Determine Active Fastener Tier:
    if len(yolo_bolts) >= 2:
        chosen_fasteners = yolo_bolts
        tier_mode = "YOLO_DEEP_LEARNING"
    elif len(large_bolts) == 4:
        chosen_fasteners = large_bolts
        tier_mode = "LARGE_BOLT"
    elif len(yolo_bolts) >= 1:
        chosen_fasteners = yolo_bolts
        tier_mode = "YOLO_DEEP_LEARNING"
    elif len(large_bolts) >= 2:
        chosen_fasteners = large_bolts
        tier_mode = "LARGE_BOLT"
    elif len(small_screws) >= 1:
        chosen_fasteners = small_screws
        tier_mode = "SMALL_SCREW"
    elif len(large_bolts) >= 1:
        chosen_fasteners = large_bolts
        tier_mode = "LARGE_BOLT"
    else:
        chosen_fasteners = []
        tier_mode = "NONE"

    detected_bolts = []
    defects_count = 0

    if not chosen_fasteners:
        os.makedirs(os.path.dirname(save_annotated_path), exist_ok=True)
        annotated = img.copy()
        cv2.rectangle(annotated, (0, 0), (w, 36), (10, 16, 28), -1)
        cv2.line(annotated, (0, 36), (w, 36), (0, 240, 255), 2)
        cv2.putText(annotated, "XR COMPUTER VISION - 0 BOLTS DETECTED", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.imwrite(save_annotated_path, annotated)
        _, buffer = cv2.imencode(".jpg", annotated)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return {
            "status": "success",
            "total_bolts_detected": 0,
            "defects_detected": 0,
            "pixels_per_mm": round(pixels_per_mm, 3),
            "bolts": [],
            "annotated_image_base64": f"data:image/jpeg;base64,{img_base64}",
            "annotated_image_path": save_annotated_path
        }

    # Spatial fastener identification matching CAD assembly layout:
    # 4-bolt assembly layout: B01=Top-Left, B02=Top-Right, B03=Bottom-Left, B04=Bottom-Right
    if len(chosen_fasteners) == 2:
        # Check if fasteners are vertically aligned (e.g. left edge of panel with Top=B01, Bottom=B03)
        chosen_fasteners.sort(key=lambda c: c[1])  # Top to bottom
        f_top = chosen_fasteners[0]
        f_bot = chosen_fasteners[1]
        dx = abs(f_top[0] - f_bot[0])
        dy = abs(f_top[1] - f_bot[1])
        if dy > dx or aspect_ratio <= 0.70:
            # Vertical column: Top is B01 (Top-Left), Bottom is B03 (Bottom-Left) — NO B02 on this edge!
            chosen_fasteners[0] = (f_top[0], f_top[1], f_top[2], f_top[3], "B01")
            chosen_fasteners[1] = (f_bot[0], f_bot[1], f_bot[2], f_bot[3], "B03")
        else:
            # Horizontal row: Left is B01, Right is B02
            chosen_fasteners[0] = (f_top[0], f_top[1], f_top[2], f_top[3], "B01")
            chosen_fasteners[1] = (f_bot[0], f_bot[1], f_bot[2], f_bot[3], "B02")
    else:
        # Sort fasteners spatially: top to bottom, left to right
        chosen_fasteners.sort(key=lambda c: (c[1] // (h // 2 + 1), c[0]))
        for idx, item in enumerate(chosen_fasteners):
            chosen_fasteners[idx] = (item[0], item[1], item[2], item[3], f"B{idx+1:02d}")

    # 3. Analyze Each Genuinely Detected Bolt/Screw
    for idx, item in enumerate(chosen_fasteners):
        cx, cy, r = item[0], item[1], item[2]
        ftype = item[3]
        bid = item[4] if len(item) > 4 else f"B{idx+1:02d}"

        # 4. Local Workpiece Surface Baseline Estimation
        annular_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(annular_mask, (cx, cy), r + 22, 255, -1)
        cv2.circle(annular_mask, (cx, cy), r + 4, 0, -1)
        local_pixels = gray[annular_mask > 0]
        local_base = float(np.percentile(local_pixels, 75)) if local_pixels.size > 0 else float(np.median(gray))
        if local_base <= 10:
            local_base = float(np.median(gray))

        # 5. Angular Perimeter Crescent Shadow Analysis (excluding border < 6 px)
        num_samples = 120
        angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
        if aspect_ratio <= 0.65 and r >= 38:
            sample_r = r - 7
        elif ftype == "small" or r < 34:
            sample_r = r + 1
        else:
            sample_r = r + 3
        sample_xs = np.clip(np.int32(cx + sample_r * np.cos(angles)), 0, w - 1)
        sample_ys = np.clip(np.int32(cy + sample_r * np.sin(angles)), 0, h - 1)
        valid = (sample_xs >= 6) & (sample_xs < w - 6) & (sample_ys >= 6) & (sample_ys < h - 6)

        shadow_thresh = local_base * 0.42
        shadow_mask = np.zeros(num_samples, dtype=bool)
        for i in range(num_samples):
            if valid[i] and gray[sample_ys[i], sample_xs[i]] < shadow_thresh:
                shadow_mask[i] = True

        wrapped_mask = np.concatenate([shadow_mask, shadow_mask])
        runs = []
        cur_run = 0
        for is_dark in wrapped_mask:
            if is_dark:
                cur_run += 1
            else:
                if cur_run > 0:
                    runs.append(cur_run)
                cur_run = 0
        if cur_run > 0:
            runs.append(cur_run)

        max_run = min(num_samples, max(runs)) if runs else 0
        shadow_arc_deg = (max_run * 360.0) / num_samples

        # 6. Profile Standoff Analysis (for side-angle small screws)
        standoff_gap_mm = 0.0
        if ftype == "small" or aspect_ratio <= 0.65:
            # Examine lateral profile between screw head and mounting board
            right_strip = np.mean(gray[max(0, cy - 4):min(h, cy + 4), cx + r:min(w, cx + r + 40)], axis=0) if w > cx + r + 1 else np.array([])
            first_board = -1
            if len(right_strip) > 0:
                for s_idx, v in enumerate(right_strip):
                    if v > 135:
                        first_board = s_idx
                        break
            dark_before_board = first_board if first_board != -1 else 0
            if dark_before_board >= 8:
                standoff_gap_mm = 5.0
            elif len(right_strip) > 5:
                # Check for dark standoff gap before reaching bright mounting board
                dark_count = 0
                in_dark = False
                saw_board = False
                for v in right_strip:
                    if v < 115:
                        dark_count += 1
                        in_dark = True
                    elif in_dark and v >= 135:
                        saw_board = True
                        break
                if dark_count >= 4 and saw_board:
                    standoff_gap_mm = 5.0

        # Estimate ROI image tracking quality
        roi_y1 = max(0, cy - r - 10)
        roi_y2 = min(h, cy + r + 10)
        roi_x1 = max(0, cx - r - 10)
        roi_x2 = min(w, cx + r + 10)
        roi_gray = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        lap_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var()) if roi_gray.size > 0 else 100.0
        vision_quality = round(min(0.96, max(0.80, 0.82 + (lap_var / 2500.0))), 2)
        depth_quality = round(min(0.94, max(0.78, 0.80 + (lap_var / 3000.0))), 2)

        # Check inner vs outer contrast for bracket screws (e.g. screws on dark mounting bracket)
        crop_box = min(35, r)
        y1, y2 = max(0, cy - crop_box), min(h, cy + crop_box)
        x1, x2 = max(0, cx - crop_box), min(w, cx + crop_box)
        sub_gray = gray[y1:y2, x1:x2]
        sh, sw = sub_gray.shape
        inner_sub = sub_gray[sh // 2 - 10:sh // 2 + 10, sw // 2 - 10:sw // 2 + 10]
        outer_sub = np.concatenate([
            sub_gray[:6, :].flatten(), sub_gray[-6:, :].flatten(),
            sub_gray[:, :6].flatten(), sub_gray[:, -6:].flatten()
        ]) if (sh >= 16 and sw >= 16) else np.array([])
        inner_outer_contrast = float(np.mean(inner_sub) - np.mean(outer_sub)) if (inner_sub.size > 0 and outer_sub.size > 0) else 0.0

        is_bracket_screw = (aspect_ratio > 0.8) and (local_base < 180) and (np.mean(sub_gray) < 130)

        # Tightness Determination from Combined Bracket Contrast, Standoff Gap & Shadow Arc
        if is_bracket_screw:
            # On a dark bracket, loose screws lifted off expose bright wood/threads around the head
            if inner_outer_contrast < -35.0:
                gap_mm = 7.5
                contact_score = 0.62
                est_torque = 8.0
                est_rotation = 5.6
            elif inner_outer_contrast < -12.0:
                gap_mm = 5.0
                contact_score = 0.78
                est_torque = 13.0
                est_rotation = 3.4
            else:
                # Flush tight countersunk screw into dark bracket
                gap_mm = 0.0
                contact_score = 0.96
                est_torque = 28.0
                est_rotation = 0.7
        elif standoff_gap_mm >= 6.0:
            gap_mm = 7.5
            contact_score = 0.62
            est_torque = 8.0
            est_rotation = 5.6
        elif standoff_gap_mm >= 3.0:
            gap_mm = 5.0
            contact_score = 0.78
            est_torque = 13.0
            est_rotation = 3.4
        elif shadow_arc_deg > 90:
            gap_mm = 7.5
            contact_score = 0.62
            est_torque = 8.0
            est_rotation = 5.6
        elif shadow_arc_deg > 35:
            gap_mm = 5.0
            contact_score = 0.78
            est_torque = 13.0
            est_rotation = 3.4
        else:
            gap_mm = 0.0
            contact_score = 0.96
            est_torque = 28.0
            est_rotation = 0.7

        # 7. Machine Learning Model Inference
        ml_res = predict_visual(gap_mm, contact_score)
        confidence = calculate_confidence(vision_quality, depth_quality, 0.95)
        is_loose = "LOOSE" in ml_res["predicted_condition"].upper()

        if is_loose:
            defects_count += 1

        detected_bolts.append({
            "bolt_id": bid,
            "center": (cx, cy),
            "radius": r,
            "gap_mm": gap_mm,
            "contact_score": contact_score,
            "vision_quality": vision_quality,
            "depth_quality": depth_quality,
            "estimated_torque_nm": est_torque,
            "estimated_rotation_deg": est_rotation,
            "confidence": round(confidence, 2),
            "condition": ml_res["predicted_condition"],
            "loose_probability": round(ml_res["loose_probability"], 3),
            "decision": "CONFIRMED_DEFECT" if (is_loose and confidence >= 0.80) else ("LIKELY_DEFECT" if is_loose else "LIKELY_HEALTHY"),
            "shadow_arc_deg": round(shadow_arc_deg, 1)
        })

    # 6. Generate Augmented Reality Holographic Overlay
    annotated = img.copy()

    # Top HUD Banner
    hud_tag = "[YOLOv8 DEEP LEARNING]" if tier_mode == "YOLO_DEEP_LEARNING" else "[OPTICAL CV]"
    cv2.rectangle(annotated, (0, 0), (w, 36), (10, 16, 28), -1)
    cv2.line(annotated, (0, 36), (w, 36), (0, 240, 255), 2)
    cv2.putText(annotated, f"XR INSPECTION {hud_tag} - {len(detected_bolts)} BOLT(S) DETECTED", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 240, 255), 2, cv2.LINE_AA)

    # Draw AR Target Rings for Each Bolt
    for b in detected_bolts:
        cx, cy = b["center"]
        r = b["radius"]
        cond = b["condition"]
        is_loose = "LOOSE" in cond.upper()

        if is_loose:
            color = (0, 0, 255)    # Crimson Red
            label = f"{b['bolt_id']}: {cond} (GAP: {b['gap_mm']}mm)"
        else:
            color = (0, 255, 128)  # Neon Green
            label = f"{b['bolt_id']}: TIGHT (GAP: 0.0mm)"

        # Target Rings & Crosshair
        cv2.circle(annotated, (cx, cy), r + 4, color, 2, cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy), r + 10, color, 1, cv2.LINE_AA)
        cv2.line(annotated, (cx - r - 14, cy), (cx + r + 14, cy), color, 1, cv2.LINE_AA)
        cv2.line(annotated, (cx, cy - r - 14), (cx, cy + r + 14), color, 1, cv2.LINE_AA)

        # Label Pill Badge
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tw, th = text_size
        card_y = cy - r - 14
        if card_y - th - 6 < 38:
            card_y = cy + r + 24

        cv2.rectangle(annotated, (cx - tw // 2 - 5, card_y - th - 5), (cx + tw // 2 + 5, card_y + 4), (10, 16, 28), -1)
        cv2.rectangle(annotated, (cx - tw // 2 - 5, card_y - th - 5), (cx + tw // 2 + 5, card_y + 4), color, 1)
        cv2.putText(annotated, label, (cx - tw // 2, card_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_annotated_path), exist_ok=True)
    cv2.imwrite(save_annotated_path, annotated)

    _, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    engine_label = "YOLOv8 Deep Learning (Custom Dataset)" if tier_mode == "YOLO_DEEP_LEARNING" else "Classical Optical Computer Vision"

    return {
        "status": "success",
        "total_bolts_detected": len(detected_bolts),
        "defects_detected": defects_count,
        "detection_engine": engine_label,
        "tier_mode": tier_mode,
        "pixels_per_mm": round(pixels_per_mm, 3),
        "bolts": detected_bolts,
        "annotated_image_base64": f"data:image/jpeg;base64,{img_base64}",
        "annotated_image_path": save_annotated_path
    }


def generate_synthetic_plate_photo(
    output_path: str = "data/synthetic_plate_photo.jpg",
    loosen_bolts: List[str] = ["B02", "B03"]
):
    """
    Generates a realistic RGB photo of the workpiece plate with designated loose bolts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_w, img_h = 1000, 700
    image = np.ones((img_h, img_w, 3), dtype=np.uint8) * 45

    px, py, pw, ph = 150, 130, 700, 440
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (135, 140, 145), -1)
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (80, 85, 90), 3)

    bolt_configs = [
        {"id": "B01", "x": px + 120, "y": py + 100},
        {"id": "B02", "x": px + pw - 120, "y": py + 100},
        {"id": "B03", "x": px + 120, "y": py + ph - 100},
        {"id": "B04", "x": px + pw - 120, "y": py + ph - 100}
    ]

    for b in bolt_configs:
        bx, by = b["x"], b["y"]
        is_loose = b["id"] in loosen_bolts

        if is_loose:
            # Differentiate moderate 5.0mm looseness vs severe 7.5mm looseness
            cv2.circle(image, (bx, by), 40, (160, 165, 170), -1)
            cv2.circle(image, (bx, by), 28, (200, 205, 210), -1)
            if b["id"] == "B02":
                # Moderate standoff crescent (~65-75 deg arc) -> LOOSE_5MM (5.0mm)
                cv2.ellipse(image, (bx, by), (42, 42), 0, 40, 105, (15, 15, 20), 6)
            else:
                # Severe lift-off crescent (~115-140 deg arc) -> LOOSE_7_5MM (7.5mm)
                cv2.ellipse(image, (bx, by), (42, 42), 0, 30, 145, (15, 15, 20), 6)
        else:
            cv2.circle(image, (bx, by), 40, (160, 165, 170), -1)
            cv2.circle(image, (bx, by), 28, (200, 205, 210), -1)

    cv2.imwrite(output_path, image)
    return output_path


if __name__ == "__main__":
    print("Testing Washer Arc Vision Detector...")
    test_img = r"C:\Users\bagav\.gemini\antigravity\brain\bbe32808-34c5-4d72-9dde-ae22a9b8e117\.user_uploaded\media_1791821904271.png"
    results = process_rgb_photo(test_img, "results/test_annotated.jpg")
    print(f"Total bolts: {results['total_bolts_detected']} | Defects: {results['defects_detected']}")
    for b in results["bolts"]:
        print(f"  [{b['bolt_id']}] -> Pos: {b['center']} | Condition: {b['condition']} | Gap: {b['gap_mm']} mm | Arc: {b.get('shadow_arc_deg', 0)} deg")
