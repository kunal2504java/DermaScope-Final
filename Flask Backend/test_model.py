import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define class names
class_names = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

def load_and_preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    try:
        # Load image using PIL
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image to 64x64
        img = img.resize((64, 64))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Normalize pixel values
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error in load_and_preprocess_image: {str(e)}")
        raise

def predict_image(model, processed_image):
    """Make predictions on a preprocessed image."""
    # Class names
    class_names = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis',
                   'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']
    
    # Get predictions
    predictions = model.predict(processed_image)
    predictions = predictions[0]
    
    # Sort predictions by probability
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_predictions = [(class_names[idx], float(predictions[idx])) for idx in sorted_idx]
    
    return sorted_predictions

def plot_prediction(image_path, predictions):
    """Plot the image and its predictions."""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot image
    img = plt.imread(image_path)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # Plot predictions
    classes = [p[0] for p in predictions]
    probabilities = [p[1] for p in predictions]
    
    # Only show top 5 predictions
    if len(predictions) > 5:
        classes = classes[:5]
        probabilities = probabilities[:5]
    
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the model
    try:
        model = load_model('my_skin_disease_pred_model.h5')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check if test_images directory exists
    if not os.path.exists('test_images'):
        print("Error: test_images directory not found")
        return

    # Get all image files from test_images directory
    image_files = [f for f in os.listdir('test_images') 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in test_images directory")
        return

    print(f"Found {len(image_files)} images to test")
    
    # Process each image
    for image_file in image_files:
        image_path = os.path.join('test_images', image_file)
        print(f"\nProcessing {image_file}...")
        
        try:
            # Load and preprocess image
            processed_image = load_and_preprocess_image(image_path)
            
            # Get predictions
            predictions = predict_image(model, processed_image)
            
            # Print predictions
            print("\nPredictions:")
            for class_name, probability in predictions[:5]:  # Show top 5 predictions
                print(f"{class_name}: {probability:.4f}")
            
            # Plot results
            plot_prediction(image_path, predictions)
            
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

if __name__ == '__main__':
    main() 