import os
import requests
import shutil
from PIL import Image
import io
import json
import zipfile

def download_ham10000():
    """Download and extract images from HAM10000 dataset."""
    try:
        # Create test directory
        os.makedirs('test_images', exist_ok=True)
        
        # HAM10000 dataset URL from Harvard Dataverse
        dataset_url = "https://dataverse.harvard.edu/api/access/datafile/3172331"
        
        print("Downloading HAM10000 dataset...")
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        
        # Save the zip file
        zip_path = "ham10000.zip"
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the first 4 images
            image_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')][:4]
            for image_file in image_files:
                zip_ref.extract(image_file, 'test_images')
                print(f"Extracted {image_file}")
        
        # Clean up
        os.remove(zip_path)
        return True
        
    except Exception as e:
        print(f"Error downloading HAM10000 dataset: {str(e)}")
        return False

if __name__ == '__main__':
    # Clean up existing test data
    if os.path.exists('test_images'):
        shutil.rmtree('test_images')
    
    if not download_ham10000():
        print("Failed to download real images. Please try again later or contact the dataset maintainers.") 