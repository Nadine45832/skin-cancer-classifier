from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH  = "skin_cancer_model/skin_cancer_resnet50.keras"
STATIC_DIR  = "static"
IMG_HEIGHT  = 224
IMG_WIDTH   = 224

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion",
]

# Load once at startup
model = tf.keras.models.load_model(MODEL_PATH)

# Serve static files (index.html, etc.) from ./static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return FileResponse(f"{STATIC_DIR}/index.html")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    tensor = preprocess(contents)

    preds = model.predict(tensor)[0]
    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(preds[top_idx])

    # Return full distribution as well
    all_classes = [
        {"label": CLASS_NAMES[i], "probability": float(preds[i])}
        for i in np.argsort(preds)[::-1]
    ]

    return {
        "prediction": top_label,
        "probability": top_prob,
        "all_classes": all_classes,
    }
