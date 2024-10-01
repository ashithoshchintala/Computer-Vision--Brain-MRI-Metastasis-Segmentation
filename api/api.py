from fastapi import FastAPI, File, UploadFile
import uvicorn
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

model = load_model('path_to_trained_model.h5')  # Load the best model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    image = image.resize((224, 224))  # Ensure the image is resized to match model input
    image = np.array(image) / 255.0
    image = image[np.newaxis, ..., np.newaxis]

    prediction = model.predict(image)

    return {"segmentation_result": prediction.tolist()}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
