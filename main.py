from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io,os

origins = [
    "http://localhost:5500",
    "http://localhost:8000",
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow specific origins
    allow_credentials=True,
    allow_methods=["POST"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)



@app.get('/')
def getHome():
    return {"message": "Hello World"}


UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


# Load the trained model
model = load_model('trained_model.keras')
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("\n\n\n\nFile: ", file)
    # Read the file content as bytes
    file_content = await file.read()
    # Convert bytes to a stream and open it with PIL
    image = Image.open(io.BytesIO(file_content))



    # file_path = os.path.join(UPLOAD_DIR, file.filename)
    # image.save(file_path)

    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values to range [0, 1]
    image_array = image_array / 255.0
    # Print the shape of the array (should be (28, 28))
    print("Shape of numpy array:", image_array.shape)
    # Predict the class
    image= image_array.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    p= np.argmax(prediction)
    print("predicted_class", int(p))
    return {"predicted_class": int(p)}

# To run the app, use: uvicorn main:app --reload
