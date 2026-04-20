from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import io
import os
import torch
from torch import nn
from torchvision import models, transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_TF = os.path.join(BASE_DIR, "skin_cancer_model", "skin_cancer_resnet50.keras")
MODEL_PATH_PT = os.path.join(BASE_DIR, "skin_cancer_model", "pytorch_skin_cancer_resnet50.pth")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMG_HEIGHT = 224
IMG_WIDTH = 224

CLASS_NAMES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesion",
]

TF_MODEL = tf.keras.models.load_model(MODEL_PATH_TF) if os.path.exists(MODEL_PATH_TF) else None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PREPROCESS_PT = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_pt_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, len(CLASS_NAMES)),
    )

    model.load_state_dict(torch.load(MODEL_PATH_PT, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

PT_MODEL = build_pt_model() if os.path.exists(MODEL_PATH_PT) else None

# Serve static files (index.html, etc.) from ./static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def root():
    return FileResponse(f"{STATIC_DIR}/index.html")


def preprocess_tf(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def preprocess_pt(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = PREPROCESS_PT(img)
    return tensor.unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = Form("tf")):
    contents = await file.read()

    if model_type == "pytorch":
        if PT_MODEL is None:
            return {"error": "PyTorch model not found on server."}

        tensor = preprocess_pt(contents).to(DEVICE)
        with torch.no_grad():
            outputs = PT_MODEL(tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        preds = probs
    else:
        if TF_MODEL is None:
            return {"error": "TensorFlow model not found on server."}

        tensor = preprocess_tf(contents)
        preds = TF_MODEL.predict(tensor)[0]

    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    top_prob = float(preds[top_idx])

    all_classes = [
        {"label": CLASS_NAMES[i], "probability": float(preds[i])}
        for i in np.argsort(preds)[::-1]
    ]

    return {
        "prediction": top_label,
        "probability": top_prob,
        "all_classes": all_classes,
        "model_type": model_type,
    }
