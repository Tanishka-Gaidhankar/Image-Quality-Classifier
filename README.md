<img width="961" height="495" alt="Screenshot 2025-08-04 174931" src="https://github.com/user-attachments/assets/993a2af6-359e-4ba1-8fd4-af72c38fd799" />
<img width="964" height="675" alt="Screenshot 2025-08-04 174939" src="https://github.com/user-attachments/assets/415a0dbc-e10a-4ca4-80b8-1ddd55458b36" />

Demo Video:demo_video.mp4

#  Image Quality Classification Model (Good vs Bad Faces)

This project aims to automatically assess the **visual quality of human face images** — especially for platforms where users upload images for verification, listing, or analysis. Poor image quality (blurry, noisy, dark, etc.) can reduce the accuracy of downstream AI models. This system classifies face images as **"Good"** or **"Bad"** based on quality metrics.

---

##  02. Dataset Used

We used the **[UTKFace dataset]** — a public dataset containing over 20,000 cropped and aligned face images, labeled with age, gender, and ethnicity.

### Dataset Preparation:
- High-quality face images from UTKFace were labeled as **"Good"**.
- We used **Albumentations** to generate synthetic **"Bad"** images by applying:
  - Motion Blur
  - Gaussian Noise
  - Over/Under Exposure
  - JPEG Compression
  - Downscaling (Low resolution)

A CSV file was created mapping each image to its label: `Good` or `Bad`.

---

## ⚙️ 03. How It Works

The system consists of:

###  1. Preprocessing
- Images resized to 128x128
- Normalized pixel values to [0,1]
- Data split into 80% training and 20% testing

###  2. Model Architecture 
A simple CNN classifier:
- Conv2D + MaxPooling
- Flatten + Dense + Dropout
- Sigmoid output for binary classification

###  3. Metrics-Based Evaluation
Alongside the classification, we also compute:
- **Clarity Score** (based on Laplacian variance)
- **Brightness Score** (mean pixel value in HSV)
- **Noise Score** (difference from Gaussian-blurred image)

These metrics help provide suggestions for how to improve a poor-quality image.
---
Run this app by running streamlit run app.py on terminal

📁 File Structure

├── good_images/       # Clean UTKFace images

├── bad_images/           # Augmented poor-quality images

├── dataset_labels.csv    # Labels (Good/Bad)

├── train_classifier.py   # Training script

├── test_model.py         # Single image test script

├── app.py                # Streamlit interface

├── good_bad_classifier.h5 # Trained model

├── README.md


