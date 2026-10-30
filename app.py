import json
from pathlib import Path

import numpy as np
import keras
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Medical Image Classifier")


@app.get("/modules")
def modules():
    return {"modules": [p.stem for p in Path("models").glob("*.keras")]}


@app.post("/predict")
async def predict(module: str = Form(...), image: UploadFile = File(...)):
    model_path = Path("models") / f"{module}.keras"
    labels_path = Path("models") / f"{module}.json"
    if not model_path.exists() or not labels_path.exists():
        raise HTTPException(404, f"No trained model found for {module}")
    try:
        raw = await image.read()
        picture = Image.open(__import__("io").BytesIO(raw)).convert("RGB").resize((224, 224))
        batch = np.expand_dims(np.asarray(picture, dtype=np.float32), 0)
        model = keras.models.load_model(model_path, compile=False)
        probabilities = model.predict(batch, verbose=0)[0]
        labels = json.loads(labels_path.read_text())
        index = int(np.argmax(probabilities))
        return {"module": module, "prediction": labels[index], "confidence": float(probabilities[index]), "scores": dict(zip(labels, probabilities.tolist()))}
    except Exception as exc:
        raise HTTPException(400, f"Could not process image: {exc}") from exc
