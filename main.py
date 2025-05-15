from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from utils import preprocess_image
import numpy as np

# Load model
model = load_model('model/FFD.keras')
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
