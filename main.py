import hmac
import os
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps
from facenet_pytorch import InceptionResnetV1, MTCNN
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("KYC_KTP_MODEL", str(ROOT / "models" / "ktp_fields.pt")))
API_KEY = os.getenv("KYC_API_KEY", "").strip()
MAX_FILE_BYTES = 7 * 1024 * 1024
FACE_THRESHOLD = float(os.getenv("KYC_FACE_THRESHOLD", "0.60"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NIK, CLASS_NAME, CLASS_FACE, CLASS_TEXT = 0, 1, 2, 3

if not MODEL_PATH.is_file():
    raise RuntimeError(f"Model KTP tidak ditemukan: {MODEL_PATH}")

app = FastAPI(title="XPDC Transporter Photo Verification", version="1.0.0")
ktp_detector = YOLO(str(MODEL_PATH))
face_detector = MTCNN(keep_all=True, device=DEVICE)
face_encoder = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
inference_lock = Lock()


def decode_image(raw: bytes) -> np.ndarray:
    try:
        image = ImageOps.exif_transpose(Image.open(BytesIO(raw))).convert("RGB")
        return np.asarray(image)
    except Exception as error:
        raise HTTPException(status_code=422, detail="File bukan gambar yang dapat dibaca.") from error


def detect_ktp_fields(image: np.ndarray) -> list[dict[str, Any]]:
    result = ktp_detector.predict(image, imgsz=1024, conf=0.15, max_det=100, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    return [
        {"class_id": int(cls), "confidence": float(conf), "box": box.astype(int).tolist()}
        for box, cls, conf in zip(boxes, classes, confidences)
        if cls in (CLASS_NIK, CLASS_NAME, CLASS_FACE, CLASS_TEXT)
    ]


def box_center(box: list[int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def find_front_ktp_cluster(fields: list[dict[str, Any]], shape: tuple[int, int]) -> dict[str, Any] | None:
    height, width = shape
    nik_rows = sorted((f for f in fields if f["class_id"] == CLASS_NIK), key=lambda f: f["confidence"], reverse=True)[:5]
    name_rows = sorted((f for f in fields if f["class_id"] == CLASS_NAME), key=lambda f: f["confidence"], reverse=True)[:5]
    text_rows = sorted((f for f in fields if f["class_id"] == CLASS_TEXT), key=lambda f: f["confidence"], reverse=True)[:8]
    face_rows = sorted((f for f in fields if f["class_id"] == CLASS_FACE), key=lambda f: f["confidence"], reverse=True)[:8]
    candidates = []

    for nik in nik_rows:
        for name in name_rows:
            for text in text_rows:
                core = [nik, name, text]
                x1 = min(row["box"][0] for row in core)
                y1 = min(row["box"][1] for row in core)
                x2 = max(row["box"][2] for row in core)
                y2 = max(row["box"][3] for row in core)
                core_w, core_h = max(1, x2 - x1), max(1, y2 - y1)
                nearby_faces = []
                for face in face_rows:
                    cx, cy = box_center(face["box"])
                    if x1 - core_w * 0.35 <= cx <= x2 + core_w * 0.35 and y1 - core_h * 0.45 <= cy <= y2 + core_h * 0.45:
                        nearby_faces.append(face)
                for face in nearby_faces:
                    group = core + [face]
                    confidence = sum(row["confidence"] for row in group) / len(group)
                    if confidence < 0.25:
                        continue
                    gx1 = max(0, min(row["box"][0] for row in group))
                    gy1 = max(0, min(row["box"][1] for row in group))
                    gx2 = min(width, max(row["box"][2] for row in group))
                    gy2 = min(height, max(row["box"][3] for row in group))
                    card_w, card_h = gx2 - gx1, gy2 - gy1
                    area_ratio = (card_w * card_h) / max(1, width * height)
                    aspect = max(card_w / max(1, card_h), card_h / max(1, card_w))
                    if card_w < width * 0.12 or card_h < height * 0.08 or not 0.018 <= area_ratio <= 0.48 or not 1.22 <= aspect <= 2.25:
                        continue
                    candidates.append({"box": [gx1, gy1, gx2, gy2], "face_box": face["box"], "confidence": confidence, "area_ratio": area_ratio, "aspect": aspect})

    return max(candidates, key=lambda candidate: candidate["confidence"], default=None)


def hand_grips_card(image: np.ndarray, card_box: list[int]) -> bool:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = card_box
    # Detected KTP fields occupy less area than the physical card, so include
    # a margin where fingers can hold the card edge.
    pad_x, pad_y = int((x2 - x1) * 0.22), int((y2 - y1) * 0.18)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
    rgb = image.copy()
    rgb.flags.writeable = False
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.40, model_complexity=0) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return False

    fingertip_ids = (4, 8, 12, 16, 20)
    for hand in result.multi_hand_landmarks:
        points = [(int(point.x * width), int(point.y * height)) for point in hand.landmark]
        inside = sum(x1 <= x <= x2 and y1 <= y <= y2 for x, y in points)
        fingertip_inside = any(x1 <= points[index][0] <= x2 and y1 <= points[index][1] <= y2 for index in fingertip_ids)
        if inside >= 2 and fingertip_inside:
            return True
    return False


def embedding(face_crop: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(face_crop, (160, 160), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 127.5) / 128.0
    with torch.no_grad():
        return face_encoder(tensor.to(DEVICE)).detach().cpu()


def get_person_face(image: np.ndarray, card_box: list[int]) -> np.ndarray | None:
    height, width = image.shape[:2]
    boxes, probabilities = face_detector.detect(Image.fromarray(image))
    if boxes is None or probabilities is None:
        return None
    x1, y1, x2, y2 = card_box
    valid = []
    for box, probability in zip(boxes, probabilities):
        if probability is None or probability < 0.90:
            continue
        fx1, fy1, fx2, fy2 = [int(value) for value in box]
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            continue
        area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
        if area < width * height * 0.006:
            continue
        valid.append((area, [max(0, fx1), max(0, fy1), min(width, fx2), min(height, fy2)]))
    if not valid:
        return None
    _, (fx1, fy1, fx2, fy2) = max(valid, key=lambda item: item[0])
    margin_x, margin_y = int((fx2 - fx1) * 0.18), int((fy2 - fy1) * 0.18)
    return image[max(0, fy1 - margin_y):min(height, fy2 + margin_y), max(0, fx1 - margin_x):min(width, fx2 + margin_x)]


def ktp_portrait(image: np.ndarray) -> np.ndarray | None:
    faces = [field for field in detect_ktp_fields(image) if field["class_id"] == CLASS_FACE]
    if not faces:
        return None
    x1, y1, x2, y2 = max(faces, key=lambda field: field["confidence"])["box"]
    height, width = image.shape[:2]
    margin_x, margin_y = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
    return image[max(0, y1 - margin_y):min(height, y2 + margin_y), max(0, x1 - margin_x):min(width, x2 + margin_x)]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "transporter-kyc"}


@app.post("/verify")
def verify_transporter_photo(
    ktp_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
    x_kyc_api_key: str = Header(default=""),
) -> dict[str, Any]:
    if not API_KEY or not hmac.compare_digest(x_kyc_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="API key verifikasi tidak valid.")

    ktp_bytes = ktp_image.file.read(MAX_FILE_BYTES + 1)
    selfie_bytes = selfie_image.file.read(MAX_FILE_BYTES + 1)
    if not ktp_bytes or not selfie_bytes:
        raise HTTPException(status_code=422, detail="Foto KTP dan foto diri wajib diunggah.")
    if len(ktp_bytes) > MAX_FILE_BYTES or len(selfie_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran setiap foto maksimal 7 MB.")
    ktp = decode_image(ktp_bytes)
    selfie = decode_image(selfie_bytes)
    if min(*ktp.shape[:2], *selfie.shape[:2]) < 240:
        raise HTTPException(status_code=422, detail="Resolusi foto terlalu kecil. Ambil ulang dengan kamera.")

    with inference_lock:
        ktp_portrait_crop = ktp_portrait(ktp)
        selfie_fields = detect_ktp_fields(selfie)
        card = find_front_ktp_cluster(selfie_fields, selfie.shape[:2])
        visible_front = card is not None
        facing_camera = bool(card and card["confidence"] >= 0.25 and 1.22 <= card["aspect"] <= 2.25)
        held = hand_grips_card(selfie, card["box"]) if card else False
        person_face = get_person_face(selfie, card["box"]) if card else None
        face_score = 0.0
        face_match = False
        if ktp_portrait_crop is not None and person_face is not None:
            score = torch.nn.functional.cosine_similarity(embedding(ktp_portrait_crop), embedding(person_face)).item()
            face_score = round(float(score), 4)
            face_match = score >= FACE_THRESHOLD

    checks = {
        "ktp_front_visible": {"passed": visible_front, "message": "Bagian depan KTP harus terlihat jelas di foto diri." if not visible_front else "Bagian depan KTP terdeteksi."},
        "ktp_facing_camera": {"passed": facing_camera, "message": "Arahkan bagian depan KTP ke kamera dan pastikan tulisan serta foto terlihat jelas." if not facing_camera else "Bagian depan, foto, dan kelompok teks KTP terlihat menghadap kamera."},
        "ktp_held_by_hand": {"passed": held, "message": "Pegang KTP di bagian tepinya agar jari terlihat menyentuh kartu." if not held else "Tangan terdeteksi memegang area KTP."},
        "face_match": {"passed": face_match, "score": face_score, "threshold": FACE_THRESHOLD, "message": "Wajah pada selfie belum cocok dengan foto pada KTP." if not face_match else "Wajah selfie cocok dengan foto pada KTP."},
    }
    reasons = [check["message"] for check in checks.values() if not check["passed"]]
    return {
        "approved": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "reasons": reasons,
        "message": "Foto lolos verifikasi." if not reasons else "Foto belum memenuhi syarat. " + " ".join(reasons),
    }
