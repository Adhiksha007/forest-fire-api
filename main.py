from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from util import preprocess_image
import numpy as np

import os
import gdown

model_path = "model/fire_model.keras"
file_id = "1CqmJjbIvV9xbj-jbImRSKMDnNgFq03dC"
url = f"https://drive.google.com/uc?id={file_id}"

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Download the model if not present
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)


# Load model
model = load_model('model/fire_model.keras')
class_names = ["No Fire", "Fire"]

# FastAPI app
app = FastAPI(title="Forest Fire Detection API")

# Prediction route
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img_array = preprocess_image(contents)
        prediction = model.predict(img_array)

        predicted_class = class_names[1] if prediction[0] > 0.5 else class_names[0]

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": float(prediction[0])
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
