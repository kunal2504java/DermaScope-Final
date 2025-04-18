import requests
import os
import sys
import socket

def test_predict_endpoint():
    # Test the root endpoint first
    root_url = "http://localhost:5000/"
    predict_url = "http://localhost:5000/predict"
    
    print("\n" + "="*50)
    print("Starting test...")
    print(f"Current directory: {os.getcwd()}")
    
    # Test root endpoint first
    try:
        print("\nTesting root endpoint...")
        root_response = requests.get(root_url)
        print(f"Root endpoint status: {root_response.status_code}")
        print(f"Root endpoint response: {root_response.text}")
    except Exception as e:
        print(f"Error testing root endpoint: {str(e)}")
        return
    
    # Test image path
    file_path = os.path.join(os.path.dirname(__file__), "test_images", "basal_cell.jpg")
    print(f"\nTest image path: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    if os.path.exists(file_path):
        print(f"File size: {os.path.getsize(file_path)} bytes")
    
    # Test predict endpoint
    try:
        print("\nOpening file...")
        with open(file_path, 'rb') as f:
            print("File opened successfully")
            files = {'file': ('basal_cell.jpg', f, 'image/jpeg')}
            print("\nSending request to predict endpoint...")
            response = requests.post(predict_url, files=files)
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            print(f"Response content: {response.text}")
            
    except requests.exceptions.ConnectionError as e:
        print("\nConnection Error:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nMake sure the Flask server is running on port 5000")
    except Exception as e:
        print("\nError:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nStack trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_predict_endpoint() 