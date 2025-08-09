import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load model
model = load_model("good_bad_classifier.h5")

# Preprocess function
def preprocess_image(img):
    img = img.resize((128, 128))  # Resize to match model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Noise detection using adaptive threshold
def is_noisy(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    # Dataset-based thresholds (tuned experimentally)
    if laplacian_var < 50:
        return "Blurry (low detail)"
    elif 50 <= laplacian_var <= 200:
        return "Clear"
    else:
        return "Noisy (too sharp / grainy)"

# Streamlit UI
st.title("Image Quality Classifier")
st.write("Upload an image to check if it's **Good** or **Bad** and get improvement suggestions.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Model prediction
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    label = "Good" if prediction[0][0] > 0.5 else "Bad"

    st.write(f"### Quality Prediction: **{label}**")

    # Noise check
    noise_status = is_noisy(img)
    st.write(f"**Noise Check:** {noise_status}")

    # Suggestions
    suggestions = []
    if label == "Bad":
        if noise_status == "Blurry (low detail)":
            suggestions.append("Increase sharpness and focus while capturing.")
        elif noise_status == "Noisy (too sharp / grainy)":
            suggestions.append("Reduce ISO or use noise reduction.")
        else:
            suggestions.append("Improve lighting and framing.")
    else:
        suggestions.append("Image looks good!")

    st.write("### Suggestions:")
    for s in suggestions:
        st.write(f"- {s}")
