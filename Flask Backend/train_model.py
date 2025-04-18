import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from main import create_model
from PIL import Image
import cv2

def is_valid_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except:
        return False

def clean_dataset(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                print(f"Removing invalid image: {file_path}")
                os.remove(file_path)

def train_model():
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Clean the dataset
    print("Cleaning training dataset...")
    clean_dataset('temp/train')
    print("Cleaning validation dataset...")
    clean_dataset('temp/validation')

    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create data generators
    batch_size = 4  # Smaller batch size for small dataset
    train_generator = train_datagen.flow_from_directory(
        'temp/train',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        'temp/validation',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Create the model
    model = create_model()

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # More patience for small dataset
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'models/skin_cancer_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Train the model
    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=100,  # More epochs for small dataset
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint]
    )

    # Save the final model
    model.save('models/skin_cancer_model_final.h5')

    # Print training results
    print("\nTraining Results:")
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    train_model() 