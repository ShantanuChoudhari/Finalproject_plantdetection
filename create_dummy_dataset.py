# create_dummy_dataset.py
import numpy as np
import cv2
import os

# Classes to simulate
classes = ["Healthy", "Early_Blight", "Late_Blight", "Powdery_Mildew"]

# Create raw_dataset structure
os.makedirs("raw_dataset", exist_ok=True)

for cls in classes:
    folder = os.path.join("raw_dataset", cls)
    os.makedirs(folder, exist_ok=True)

    # Generate 10 dummy images per class
    for i in range(10):
        # Random 128x128 RGB image
        img = np.random.randint(0, 255, (128,128,3), dtype=np.uint8)
        filename = os.path.join(folder, f"{cls}_{i}.jpg")
        cv2.imwrite(filename, img)

print("✅ Dummy dataset created inside 'raw_dataset/' folder")
