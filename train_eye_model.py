import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "eye_dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "eye_dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
print("Classes:", class_names)

# Save class names to file (VERY IMPORTANT)
os.makedirs("models", exist_ok=True)
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

model.save("models/eye_model.h5")

print("Eye Model Saved Successfully!")
