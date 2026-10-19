import urllib.request
import json

boundary = "boundary123"

def test_photo(img_path, label):
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        b'Content-Disposition: form-data; name="file"; filename="upload.jpg"\r\n',
        b"Content-Type: image/jpeg\r\n\r\n",
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8")
    ]
    body = b"".join(parts)

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/inspect/photo",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )

    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        print(f"\n=== {label} ===")
        print(f"Total Bolts Detected: {res['total_bolts_detected']} | Defects: {res['defects_detected']}")
        for b in res["bolts"]:
            print(f"  [{b['bolt_id']}] -> Center: {b['center']} | Radius: {b['radius']} | Condition: {b['condition']} | Gap: {b['gap_mm']} mm")

test_photo(
    r"C:\Users\bagav\.gemini\antigravity\brain\bbe32808-34c5-4d72-9dde-ae22a9b8e117\.user_uploaded\media_1791821904271.png",
    "TEST 1: INDUSTRIAL METAL HOLED PLATE (4 M10 BOLTS)"
)

test_photo(
    r"C:\Users\bagav\.gemini\antigravity\brain\bbe32808-34c5-4d72-9dde-ae22a9b8e117\.user_uploaded\media_1791906562126_29d334c7.jpg",
    "TEST 2: WOOD PANEL / FURNITURE (2 SMALL PROFILE SCREWS)"
)

test_photo(
    r"D:\XR_inspection\results\last_uploaded_raw.jpg",
    "TEST 3: WALL SWITCHBOARD (2 FRONTAL SCREWS - USER PHOTO)"
)
