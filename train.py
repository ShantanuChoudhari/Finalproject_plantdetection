# train.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_cnn
from utils import save_class_indices
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--model_out", type=str, default="models/plant_disease_model.h5")
    return p.parse_args()

def main():
    args = parse_args()
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")  # optional but recommended

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = train_gen.num_classes
    print(f"Detected {num_classes} classes: {train_gen.class_indices}")

    # Build model
    model = build_cnn(input_shape=(args.img_size, args.img_size, 3), num_classes=num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    checkpoint = ModelCheckpoint(args.model_out, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint, early, reduce_lr]
    )

    # Save class mapping
    save_class_indices(train_gen.class_indices, path=os.path.join(os.path.dirname(args.model_out), "class_indices.json"))
    print("Saved class indices mapping.")

    # Optionally: save final model again (best one saved by checkpoint already)
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

    # Plot training history
    plt.figure(figsize=(8,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig('models/accuracy_plot.png')

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('models/loss_plot.png')

if __name__ == "__main__":
    main()
