import os
import shutil
from sklearn.model_selection import train_test_split

def organize_dataset(source_dir, dest_dir, test_size=0.2):
    """Organize dataset into train/validation splits"""
    
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Get all images in class
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split into train/val
        train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)
        
        # Create directories
        train_class_dir = os.path.join(dest_dir, 'train', class_name)
        val_class_dir = os.path.join(dest_dir, 'val', class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy files
        for img in train_images:
            shutil.copy2(os.path.join(class_path, img), train_class_dir)
        
        for img in val_images:
            shutil.copy2(os.path.join(class_path, img), val_class_dir)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")

# Usage
# organize_dataset('raw_plantvillage', 'organized_plantvillage')