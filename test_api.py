import requests
import time

def test_api():
    url = "http://192.168.0.121:8001/api/generate-and-sync"
    data = {"text": "Hello there! This is a test."}
    
    print("Testing API endpoint...")
    print(f"URL: {url}")
    print(f"Data: {data}")
    
    try:
        response = requests.post(url, data=data, timeout=120)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        if response.status_code == 200:
            # Save the video file
            filename = f"api_test_result_{int(time.time())}.mp4"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ Success! Video saved as: {filename}")
            print(f"File size: {len(response.content)} bytes")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()
