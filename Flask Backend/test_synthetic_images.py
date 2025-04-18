import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Define class names
class_names = ['melanoma', 'nevus', 'basal cell carcinoma', 'actinic keratosis', 
               'squamous cell carcinoma', 'pigmented benign keratosis', 'seborrheic keratosis']

def main():
    # Load the saved model
    print("Loading saved model...")
    model = load_model('my_skin_disease_pred_model.h5')
    print("\nModel loaded successfully!")
    
    # Directory containing test images
    test_dir = 'test_images'
    
    # Process each image in the test directory
    for filename in os.listdir(test_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, filename)
            print(f"\nTesting {filename}:")
            
            # Load and preprocess image
            img = load_img(image_path, target_size=(64, 64))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # Normalize pixel values
            processed_image = img_array / 255.0
            
            # Make prediction
            predictions = model.predict(processed_image)[0]
            sorted_indices = np.argsort(predictions)[::-1]
            predictions = [(class_names[i], float(predictions[i])) for i in sorted_indices]
            
            # Print top predictions
            print("\nTop predictions:")
            for class_name, prob in predictions:
                print(f"{class_name}: {prob:.4f}")
            
            # Plot results
            plt.figure(figsize=(15, 5))
            
            # Plot input image
            plt.subplot(1, 2, 1)
            img = plt.imread(image_path)
            plt.imshow(img)
            plt.title('Input Image')
            plt.axis('off')
            
            # Plot predictions
            plt.subplot(1, 2, 2)
            classes = [pred[0] for pred in predictions[:5]]  # Show top 5 predictions
            probs = [pred[1] for pred in predictions[:5]]
            plt.barh(classes, probs)
            plt.title('Top 5 Prediction Probabilities')
            plt.xlabel('Probability')
            plt.xlim(0, 1)
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()