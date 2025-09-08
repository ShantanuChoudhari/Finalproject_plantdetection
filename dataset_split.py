# dataset_split.py
import os
import shutil
import random

# Path to your raw PlantVillage dataset (all images in one folder OR in class folders)
RAW_DATASET_DIR = "raw_dataset"   # put your original dataset here
OUTPUT_DIR = "dataset"

# Classes (must match the disease categories you want to use)
CLASSES = ["Healthy", "Early_Blight", "Late_Blight", "Powdery_Mildew"]

# Train/Val/Test split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_dataset():
    for cls in CLASSES:
        cls_folder = os.path.join(RAW_DATASET_DIR, cls)
        images = os.listdir(cls_folder)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_split * n_total)
        n_val = int(val_split * n_total)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        # Create output folders
        for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            create_dir(out_dir)
            for img in split_imgs:
                src = os.path.join(cls_folder, img)
                dst = os.path.join(out_dir, img)
                shutil.copy(src, dst)

    print("✅ Dataset split completed!")
    print(f"Train/Val/Test saved inside '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    split_dataset()
