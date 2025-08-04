import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 📥 Load dataset labels
df = pd.read_csv("dataset_labels.csv")
df = df[df['label'].isin(['Good', 'Bad'])]

# 🔁 Map labels to 0 and 1
label_map = {'Good': 1, 'Bad': 0}
df['label'] = df['label'].map(label_map)

# 🖼️ Load images and resize
IMG_SIZE = 128
images = []
labels = []

print("🔄 Loading images...")
for i, row in df.iterrows():
    img = cv2.imread(row['image_path'])
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    images.append(img)
    labels.append(row['label'])

images = np.array(images)
labels = np.array(labels)

# 🔀 Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 🧠 Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 🏋️ Train
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 📉 Plot training results
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()
plt.show()

# 💾 Save model
model.save("good_bad_classifier.h5")
print("✅ Model saved as good_bad_classifier.h5")
