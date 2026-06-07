# -*- coding: utf-8 -*-

import os
import uuid
import shutil
import time

import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

label_descriptions = {
    "30C": "Wash at maximum 30°C.",
    "40C": "Wash at maximum 40°C.",
    "60C": "Wash at maximum 60°C.",
    "95C": "Wash at maximum 95°C.",
    "DN_Bleach": "Do not bleach.",
    "DN_dry_clean": "Do not dry clean.",
    "DN_iron": "Do not iron.",
    "DN_tumble_dry": "Do not tumble dry.",
    "DN_wash": "Do not wash.",
    "DN_wring": "Do not wring.",
    "Drip_dry": "Drip dry.",
    "Dry_clean_any_solvent_execpt_tricholoroethylene": "Dry clean with any solvent except trichloroethylene.",
    "Dry_flat": "Dry flat.",
    "hand_wash": "Hand wash only.",
    "iron_high": "Iron at high temperature.",
    "iron_low": "Iron at low temperature.",
    "iron_medium": "Iron at medium temperature.",
    "line_dry": "Line dry.",
    "machine_wash_normal": "Machine wash normal.",
    "machine_wash_permanent_press": "Machine wash permanent press.",
    "non_chlorine_bleach": "Use non-chlorine bleach only.",
    "tumble_dry_low": "Tumble dry at low temperature.",
    "tumble_dry_medium": "Tumble dry at medium temperature.",
    "tumble_dry_normal": "Tumble dry normal."
}


@app.get("/")
def home():
    return {"message": "Care Label Detection API is running."}


def resize_image_for_inference(input_path, output_path, max_size=960):
    image = cv2.imread(input_path)

    if image is None:
        return input_path

    h, w = image.shape[:2]
    largest_side = max(h, w)

    if largest_side <= max_size:
        cv2.imwrite(output_path, image)
        return output_path

    scale = max_size / largest_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return output_path


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    start_time = time.time()

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]

    if file_extension == "":
        file_extension = ".jpg"

    original_filename = f"{file_id}_original{file_extension}"
    resized_filename = f"{file_id}_resized.jpg"
    result_filename = f"{file_id}_result.jpg"

    original_path = os.path.join(UPLOAD_DIR, original_filename)
    resized_path = os.path.join(UPLOAD_DIR, resized_filename)
    result_path = os.path.join(RESULT_DIR, result_filename)

    with open(original_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    inference_path = resize_image_for_inference(
        input_path=original_path,
        output_path=resized_path,
        max_size=960
    )

    results = model.predict(
        source=inference_path,
        imgsz=512,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]
    detections = []

    image = cv2.imread(inference_path)

    if result.boxes is not None and image is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            label = model.names[class_id]
            description = label_descriptions.get(label, "No description available.")

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "description": description
            })

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            text = f"{label} {confidence:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(image, (x1, max(y1 - 22, 0)), (x1 + len(text) * 9, y1), (255, 0, 0), -1)
            cv2.putText(
                image,
                text,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    if image is not None:
        cv2.imwrite(result_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])

    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host")
    base_url = f"{forwarded_proto}://{host}"

    elapsed_time = round(time.time() - start_time, 2)

    return {
        "detections": detections,
        "result_image_url": f"{base_url}/result/{result_filename}",
        "processing_time_seconds": elapsed_time
    }


@app.get("/result/{image_name}")
def get_result_image(image_name: str):
    image_path = os.path.join(RESULT_DIR, image_name)

    if not os.path.exists(image_path):
        return {"error": "Result image not found."}

    return FileResponse(image_path)