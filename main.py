from fastapi import FastAPI, UploadFile, File
import uvicorn
from helpers import read_file_as_image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_models("../models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Health"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    return 

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)