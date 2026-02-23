import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("models/eye_model.h5")

IMG_SIZE = 224

img = cv2.imread("eye_dataset/open/12.jpg")
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
print("Prediction:", prediction)
