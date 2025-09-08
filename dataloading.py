import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
import streamlit as st

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class PlantDiseaseDetector:
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.model = None
        self.class_names = None
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the PlantVillage dataset"""
        # Create datasets
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
        
        self.class_names = train_ds.class_names
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Normalize pixel values to [0,1]
        normalization_layer = layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def create_model(self, num_classes):
        """Create CNN model architecture"""
        model = keras.Sequential([
            # Data augmentation layers
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Convolutional layers
            layers.Conv2D(32, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, 3, activation='relu'),
            layers.MaxPooling2D(),
            
            # Classification head
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, train_ds, val_ds, epochs=25):
        """Train the model with callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint('best_plant_model.h5', save_best_only=True)
        ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, val_ds):
        """Evaluate model performance"""
        # Get predictions
        y_pred = []
        y_true = []
        
        for images, labels in val_ds:
            predictions = self.model.predict(images)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_true, y_pred
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_disease(self, image_path):
        """Predict disease from single image"""
        # Load and preprocess image
        img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array /= 255.0  # Normalize
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.class_names[predicted_class], confidence

# Example usage for training
def train_plant_disease_model():
    # Initialize detector
    detector = PlantDiseaseDetector()
    
    # Load data (adjust path to your PlantVillage dataset)
    data_dir = "path/to/plantvillage/dataset"  # Update this path
    train_ds, val_ds = detector.load_and_preprocess_data(data_dir)
    
    # Create and train model
    num_classes = len(detector.class_names)
    model = detector.create_model(num_classes)
    
    print("Model Architecture:")
    model.summary()
    
    # Train model
    print("Starting training...")
    history = detector.train_model(train_ds, val_ds, epochs=25)
    
    # Evaluate model
    print("Evaluating model...")
    y_true, y_pred = detector.evaluate_model(val_ds)
    
    # Plot results
    detector.plot_training_history(history)
    
    # Save model
    model.save('plant_disease_model.h5')
    print("Model saved as 'plant_disease_model.h5'")
    
    return detector

if __name__ == "__main__":
    # Run training
    detector = train_plant_disease_model()