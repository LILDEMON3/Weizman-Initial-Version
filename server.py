from io import BytesIO
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import numpy as np
import tensorflow as tf
import uvicorn
import os

# Define confidence threshold (e.g., 50%)
CONFIDENCE_THRESHOLD = 0.5

# Define whether each label is cancerous or not
cancerous_labels = {
    'actinic keratoses': 'non-cancerous',
    'basal cell carcinoma': 'cancerous',
    'benign keratosis': 'non-cancerous',
    'dermatofibroma': 'non-cancerous',
    'melanoma': 'cancerous',
    'nevus': 'non-cancerous',
    'vascular lesions': 'non-cancerous'
}

# Mapping of class indices to actual skin cancer types (from HAM10000 metadata)
labels = list(cancerous_labels.keys())

# Preprocess image to feed into model
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
if not os.path.exists('skin_cancer_detector.h5'):
    raise FileNotFoundError("Trained model not found! Please train the model first.")

model = tf.keras.models.load_model('skin_cancer_detector.h5')

# Run model prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    processed_image = preprocess_image(image)

    # Get predictions
    predictions = model.predict(processed_image)
    max_confidence = np.max(predictions)
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    if max_confidence >= CONFIDENCE_THRESHOLD:
        predicted_label = labels[predicted_class_index]
        cancer_status = cancerous_labels[predicted_label]
        return {"match": True, "class": predicted_label, "cancerous": cancer_status, "confidence": float(max_confidence)}
    else:
        return {"match": False, "class": None, "cancerous": None, "confidence": float(max_confidence)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
