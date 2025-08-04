import os
import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm
print("Script started...")

# Input/output directories
GOOD_DIR = "good_images"
BAD_DIR = "bad_images"
CSV_PATH = "dataset_labels.csv"

# Create output folder if it doesn't exist
os.makedirs(BAD_DIR, exist_ok=True)

# Albumentations transformation to create synthetic "Bad" images
bad_transform = A.Compose([
    A.MotionBlur(p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Downscale(scale_min=0.3, scale_max=0.7, p=0.5),
    A.ImageCompression(quality_lower=5, quality_upper=30, p=0.5)
])

# Store image paths and labels
data = []

# Process all Good images
for img_name in tqdm(os.listdir(GOOD_DIR)):
    img_path = os.path.join(GOOD_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    # Save the original Good image path and label
    data.append([img_path, "Good"])

    # Apply bad transformation
    augmented = bad_transform(image=image)["image"]

    # Save Bad image
    bad_img_name = f"bad_{img_name}"
    bad_img_path = os.path.join(BAD_DIR, bad_img_name)
    cv2.imwrite(bad_img_path, augmented)

    # Save Bad label
    data.append([bad_img_path, "Bad"])

# Save labels to CSV
df = pd.DataFrame(data, columns=["image_path", "label"])
df.to_csv(CSV_PATH, index=False)

print(f"\n✅ Done. Labels saved to {CSV_PATH}")
