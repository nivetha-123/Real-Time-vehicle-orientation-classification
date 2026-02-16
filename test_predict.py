import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image

IMG_SIZE = 224

model = keras.models.load_model("orientation_model_clean.h5")

class_names = [
    "front",
    "frontleft",
    "frontright",
    "rear",
    "rearleft",
    "rearright"
]

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = np.max(score)

    print("Prediction:", predicted_class)
    print("Confidence:", float(confidence))

# Example
predict_image("samples.jpg")
