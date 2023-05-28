from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO 
from PIL import Image
import tensorflow as tf

app = FastAPI()


#Model Building
Model = tf.keras.models.load_model("/models/1")
CLASS_NAMES = ['Early blight', 'Late Blight', 'Healthy']

@app.get("/ping")
async def ping():
    return "Hello I'm Alive!"


def read_file_as_image(data) -> np.ndarray:
      image = np.array(Image.open(BytesIO(data)))      
      return image
      

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        prediction = Model.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = np.max(prediction[0])
        return {'Class': predicted_class, 'Confidence': float(confidence)}



if __name__ == '__main__':
      uvicorn.run(app, host='localhost', port=8000)

