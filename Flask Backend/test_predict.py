import requests
import os
import json

def test_predict_endpoint():
    # URL of the predict endpoint
    url = 'http://localhost:5000/predict'
    
    # Path to the test image
    image_path = 'test_images/basal_cell.jpg'
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    print(f"Testing predict endpoint with image: {image_path}")
    print(f"Image size: {os.path.getsize(image_path)} bytes")
    
    try:
        # Open and send the image file
        with open(image_path, 'rb') as image_file:
            files = {'file': image_file}
            print("Sending request to server...")
            response = requests.post(url, files=files)
            
            # Print response details
            print(f"\nResponse Status Code: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            try:
                response_json = response.json()
                print(f"Response Content: {json.dumps(response_json, indent=2)}")
                
                if response.status_code == 500:
                    print("\nServer Error Details:")
                    if 'error' in response_json:
                        print(f"Error message: {response_json['error']}")
                    if 'traceback' in response_json:
                        print(f"Traceback: {response_json['traceback']}")
            except json.JSONDecodeError:
                print(f"Raw Response Content: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Flask server is running.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    test_predict_endpoint() 