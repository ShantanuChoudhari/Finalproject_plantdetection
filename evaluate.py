# evaluate.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from utils import load_class_indices
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    p.add_argument("--model_path", type=str, default="models/plant_disease_model.h5")
    p.add_argument("--img_size", type=int, default=128)
    return p.parse_args()

def main():
    args = parse_args()
    test_dir = os.path.join(args.data_dir, "test")
    model = load_model(args.model_path)
    _, inv_class_map = load_class_indices(path=os.path.join(os.path.dirname(args.model_path), "class_indices.json"))

    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(test_dir, target_size=(args.img_size, args.img_size),
                                           batch_size=32, class_mode='categorical', shuffle=False)

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    labels = [inv_class_map[i] for i in range(len(inv_class_map))]

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/confusion_matrix.png")
    print("Saved confusion matrix to models/confusion_matrix.png")

if __name__ == "__main__":
    main()
