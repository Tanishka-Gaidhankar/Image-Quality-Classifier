import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

model = load_model("good_bad_classifier.h5")


IMG_SIZE = 128


def preprocess_image(image_path):
    print(f"Trying to read image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Could not read image at {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img


def test_image(image_path):
    img = preprocess_image(image_path)
    pred = model.predict(img)[0][0]
    label = "Good" if pred >= 0.5 else "Bad"
    print(f"✅ Prediction: {label} (Confidence: {pred:.2f})")
    image_path = os.path.join("good_images", "45_0_1_20170117180926369.jpg")
    test_image(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py path_to_image.jpg")
    else:
        test_image(sys.argv[1])
