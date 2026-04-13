import json
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import load_model, predict

MODELS_DIR       = Path("models")
MODEL_NAME       = "convnext_tiny"
WEIGHTS_PATH     = MODELS_DIR / f"{MODEL_NAME}.pth"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

model       = None
class_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names
    
    model = load_model(MODEL_NAME, str(WEIGHTS_PATH))
    
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    
    print(f"Model {MODEL_NAME} is loaded. Classes num: {len(class_names)}")
    
    yield

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict(image_bytes, model, class_names)
    return JSONResponse(content=result)

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.get("/research")
async def research():
    html = Path("static/research.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)
