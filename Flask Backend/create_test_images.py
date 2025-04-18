import numpy as np
import cv2
import os

def create_circular_lesion(size, center, radius, color):
    """Create a circular lesion with the specified parameters."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, center, radius, color, -1)
    return img

def create_irregular_lesion(size, center, radius, color, irregularity=0.3):
    """Create an irregularly shaped lesion."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    points = []
    for angle in range(0, 360, 10):
        r = radius * (1 + irregularity * np.random.random())
        x = int(center[0] + r * np.cos(np.radians(angle)))
        y = int(center[1] + r * np.sin(np.radians(angle)))
        points.append([x, y])
    points = np.array(points, np.int32)
    cv2.fillPoly(img, [points], color)
    return img

def create_melanoma_image():
    """Create a synthetic melanoma image."""
    size = 224
    center = (size//2, size//2)
    base_color = (50, 50, 50)  # Dark brown/black
    img = create_irregular_lesion(size, center, 80, base_color, irregularity=0.4)
    
    # Add some variation in color
    for i in range(3):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if img[y, x, 0] > 0:  # If it's part of the lesion
            color = (np.random.randint(30, 70), 
                    np.random.randint(30, 70), 
                    np.random.randint(30, 70))
            cv2.circle(img, (x, y), 5, color, -1)
    
    return img

def create_nevus_image():
    """Create a synthetic nevus image."""
    size = 224
    center = (size//2, size//2)
    base_color = (100, 100, 100)  # Light brown
    img = create_circular_lesion(size, center, 60, base_color)
    
    # Add some texture
    for i in range(5):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if img[y, x, 0] > 0:  # If it's part of the lesion
            color = (np.random.randint(80, 120), 
                    np.random.randint(80, 120), 
                    np.random.randint(80, 120))
            cv2.circle(img, (x, y), 3, color, -1)
    
    return img

def create_basal_cell_image():
    """Create a synthetic basal cell carcinoma image."""
    size = 224
    center = (size//2, size//2)
    base_color = (150, 150, 150)  # Pinkish
    img = create_irregular_lesion(size, center, 70, base_color, irregularity=0.2)
    
    # Add some texture and blood vessels
    for i in range(10):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if img[y, x, 0] > 0:  # If it's part of the lesion
            color = (np.random.randint(130, 170), 
                    np.random.randint(130, 170), 
                    np.random.randint(130, 170))
            cv2.circle(img, (x, y), 2, color, -1)
    
    return img

def create_keratosis_image():
    """Create a synthetic actinic keratosis image."""
    size = 224
    center = (size//2, size//2)
    base_color = (180, 180, 180)  # Light pink
    img = create_irregular_lesion(size, center, 65, base_color, irregularity=0.1)
    
    # Add some texture
    for i in range(8):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        if img[y, x, 0] > 0:  # If it's part of the lesion
            color = (np.random.randint(160, 200), 
                    np.random.randint(160, 200), 
                    np.random.randint(160, 200))
            cv2.circle(img, (x, y), 4, color, -1)
    
    return img

def main():
    # Create test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    
    # Generate and save test images
    print("Creating synthetic test images...")
    
    melanoma_img = create_melanoma_image()
    cv2.imwrite('test_images/melanoma.jpg', melanoma_img)
    print("Created test_images/melanoma.jpg")
    
    nevus_img = create_nevus_image()
    cv2.imwrite('test_images/nevus.jpg', nevus_img)
    print("Created test_images/nevus.jpg")
    
    basal_cell_img = create_basal_cell_image()
    cv2.imwrite('test_images/basal_cell.jpg', basal_cell_img)
    print("Created test_images/basal_cell.jpg")
    
    keratosis_img = create_keratosis_image()
    cv2.imwrite('test_images/keratosis.jpg', keratosis_img)
    print("Created test_images/keratosis.jpg")
    
    print("Test images created successfully!")

if __name__ == "__main__":
    main() 