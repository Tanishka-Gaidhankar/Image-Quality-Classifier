import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("good_bad_classifier.h5")
IMG_SIZE = 128

# --------------------------------------------
# 📊 Image Quality Metric Functions
# --------------------------------------------

def clarity_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return min(score / 100.0, 1.0) * 100

def brightness_score(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    brightness = hsv[..., 2].mean()
    return min(brightness / 255.0, 1.0) * 100

def noise_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = np.std(gray - blur)
    return min(noise / 50.0, 1.0) * 100

def get_suggestions(clarity, brightness, noise):
    suggestions = []
    if clarity < 50:
        suggestions.append("🔍 Image is blurry. Try using better focus or steadier capture.")
    if brightness < 40:
        suggestions.append("💡 Lighting is poor. Capture in brighter conditions.")
    if noise > 60:
        suggestions.append("📸 Image is noisy. Avoid low-light environments.")
    if not suggestions:
        suggestions.append("✅ Image quality looks great!")
    return suggestions

# --------------------------------------------
# 🖼️ Streamlit App UI
# --------------------------------------------

st.set_page_config(page_title="Image Quality Classifier", page_icon="🧠")
st.title("🧠 Image Quality Classifier")
st.markdown("Upload an image to classify it as **Good** or **Bad**, and get improvement tips.")

uploaded_file = st.file_uploader("📁 Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to OpenCV
    opencv_image = np.array(image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Preprocess for model
    model_input = cv2.resize(opencv_image, (IMG_SIZE, IMG_SIZE))
    model_input = model_input / 255.0
    model_input = np.expand_dims(model_input, axis=0)

    # Predict Good/Bad
    pred = model.predict(model_input)[0][0]
    label = "Good" if pred >= 0.5 else "Bad"
    confidence = pred if pred >= 0.5 else 1 - pred

    # Compute scores
    clarity = clarity_score(opencv_image)
    brightness = brightness_score(opencv_image)
    noise = noise_score(opencv_image)
    suggestions = get_suggestions(clarity, brightness, noise)

    # Output
    #st.subheader(f"📌 Prediction: **{label.upper()}** ({confidence:.2f} confidence)")
    st.markdown("### 🔬 Quality Metrics")
    st.progress(int(clarity), text=f"Clarity: {clarity:.1f}%")
    st.progress(int(brightness), text=f"Brightness: {brightness:.1f}%")
    st.progress(int(noise), text=f"Noise Level: {noise:.1f}%")

    st.markdown("### 💡 Suggestions")
    for s in suggestions:
        st.markdown(f"- {s}")
