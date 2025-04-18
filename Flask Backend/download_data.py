import os
import requests
import zipfile
import shutil
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def setup_sample_data():
    # Create directories
    os.makedirs('temp/train', exist_ok=True)
    os.makedirs('temp/validation', exist_ok=True)
    
    # Sample images for each class (using Kaggle dataset URLs)
    sample_images = {
        'Melanocytic nevi': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/melanocytic_nevi/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/melanocytic_nevi/2.jpg'
        ],
        'Melanoma': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/melanoma/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/melanoma/2.jpg'
        ],
        'Benign keratosis-like lesions': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/benign_keratosis/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/benign_keratosis/2.jpg'
        ],
        'Basal cell carcinoma': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/basal_cell_carcinoma/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/basal_cell_carcinoma/2.jpg'
        ],
        'Actinic keratoses': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/actinic_keratoses/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/actinic_keratoses/2.jpg'
        ],
        'Vascular lesions': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/vascular_lesions/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/vascular_lesions/2.jpg'
        ],
        'Dermatofibroma': [
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/dermatofibroma/1.jpg',
            'https://raw.githubusercontent.com/skin-cancer-detection-dataset/skin-cancer-dataset/main/dermatofibroma/2.jpg'
        ]
    }

    print("Downloading sample images...")
    
    # Download and organize training images
    for class_name, urls in sample_images.items():
        class_dir = os.path.join('temp/train', class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i, url in enumerate(urls):
            filename = os.path.join(class_dir, f'image_{i}.jpg')
            try:
                download_file(url, filename)
                print(f"Downloaded {filename}")
                
                # Copy first image to validation set
                if i == 0:
                    val_dir = os.path.join('temp/validation', class_name)
                    os.makedirs(val_dir, exist_ok=True)
                    shutil.copy2(filename, os.path.join(val_dir, f'val_image_{i}.jpg'))
            except Exception as e:
                print(f"Error downloading {url}: {str(e)}")

    print("Sample data setup complete!")

if __name__ == '__main__':
    # Clean up existing data
    if os.path.exists('temp'):
        shutil.rmtree('temp')
    
    setup_sample_data() 