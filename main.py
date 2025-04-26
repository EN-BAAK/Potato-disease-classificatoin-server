from fastapi import FastAPI, UploadFile, File
import uvicorn
from helpers import read_file_as_image
import tensorflow as tf
from globalVars import CLASS_NAMES, MODEL_LOCATION, PORT, HOST

app = FastAPI()

MODEL = tf.keras.models.load_models(MODEL_LOCATION)

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
    uvicorn.run(app, host=HOST, port=PORT)