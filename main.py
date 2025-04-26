from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from helpers import read_file_as_image
from globalVars import CLASS_NAMES, MODEL_LOCATION, PORT, HOST, ORIGINS_CORS
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
MODEL = tf.saved_model.load(MODEL_LOCATION)
infer = MODEL.signatures['serving_default']

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS_CORS, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        image = image.astype(np.float32)
        image_batch = np.expand_dims(image, 0)
        
        predictions = infer(tf.convert_to_tensor(image_batch))
        predicted_class = CLASS_NAMES[np.argmax(predictions['output_0'][0])]
        confidence = np.max(predictions['output_0'][0])

        return {
            "class": predicted_class,
            "confidence": float(confidence)
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)