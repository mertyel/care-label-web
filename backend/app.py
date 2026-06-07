# -*- coding: utf-8 -*-

import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Frontend Vercel'de, backend Render'da çalışacağı için CORS açık olmalı
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

# Modeli uygulama başlarken bir kere yüklüyoruz
model = YOLO(MODEL_PATH)

# Labels dosyasını okuyoruz
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]

# Kullanıcıya gösterilecek açıklamalar
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


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Dosya adını güvenli ve benzersiz yapıyoruz
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]

    if file_extension == "":
        file_extension = ".jpg"

    input_filename = f"{file_id}{file_extension}"
    input_path = os.path.join(UPLOAD_DIR, input_filename)

    # Yüklenen fotoğrafı kaydet
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # YOLO tahmini
    results = model.predict(
        source=input_path,
        imgsz=640,
        conf=0.20,
        save=True,
        project=RESULT_DIR,
        name=file_id,
        exist_ok=True,
        verbose=False
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
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


    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host")
    base_url = f"{forwarded_proto}://{host}"

    return {
        "detections": detections,
        "result_image_url": f"{base_url}/result/{file_id}/{input_filename}"
    }


@app.get("/result/{folder_name}/{image_name}")
def get_result_image(folder_name: str, image_name: str):
    image_path = os.path.join(RESULT_DIR, folder_name, image_name)

    if not os.path.exists(image_path):
        return {"error": "Result image not found."}

    return FileResponse(image_path)