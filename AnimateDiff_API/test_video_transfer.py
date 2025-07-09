#!/usr/bin/env python3
"""
Test script for video transfer functionality
Run this to test the video generation and transfer to main system
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8002"
API_KEY = "shashank_ka_vision786"

def test_health_check():
    """Test if the API is running"""
    print("🔍 Testing API health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy!")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        return False

def test_video_generation_with_transfer():
    """Test video generation with automatic transfer"""
    print("\n🎬 Testing video generation with transfer...")
    
    payload = {
        "prompt": "a beautiful anime girl walking in a garden, masterpiece",
        "negative_prompt": "blurry, low quality, deformed",
        "seed": 123,
        "guidance_scale": 8.0,
        "steps": 15,  # Reduced for faster testing
        "num_frames": 16,  # Reduced for faster testing
        "fps": 8,
        "subject": "Test Video",
        "topic": "API Test"
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        print("📤 Sending generation request...")
        start_time = time.time()
        
        response = requests.post(
            f"{API_BASE_URL}/generate-video-with-transfer",
            json=payload,
            headers=headers,
            timeout=600  # 10 minutes timeout
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Request completed in {elapsed_time:.1f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Video generation successful!")
            print(f"📁 Local path: {result.get('local_path')}")
            print(f"📤 Transfer success: {result.get('transfer_success')}")
            
            if result.get('transfer_success'):
                print(f"🎬 Video ID: {result.get('video_id')}")
                print(f"🔗 Access URL: {result.get('access_url')}")
            else:
                print(f"⚠️ Transfer failed: {result.get('transfer_error')}")
            
            return True
        else:
            print(f"❌ Generation failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this is normal for video generation")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_manual_transfer():
    """Test manual video transfer (if you have an existing video file)"""
    print("\n📤 Testing manual video transfer...")
    
    # This would require an existing video file
    video_path = "outputs/animation_test.mp4"  # Adjust path as needed
    
    payload = {
        "video_path": video_path,
        "subject": "Manual Test",
        "topic": "Manual Transfer Test",
        "prompt": "Test video for manual transfer"
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/send-video-to-main",
            json=payload,
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Manual transfer successful!")
            print(f"🎬 Video ID: {result.get('video_id')}")
            print(f"🔗 Access URL: {result.get('access_url')}")
            return True
        else:
            print(f"❌ Manual transfer failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Manual transfer failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting AnimateDiff Video Transfer Tests")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ API is not running. Please start the server first.")
        return
    
    # Test 2: Video generation with transfer
    print("\n" + "=" * 50)
    test_video_generation_with_transfer()
    
    # Test 3: Manual transfer (optional)
    print("\n" + "=" * 50)
    print("ℹ️ Manual transfer test skipped (requires existing video file)")
    # Uncomment the line below if you have a video file to test with
    # test_manual_transfer()
    
    print("\n" + "=" * 50)
    print("🏁 Tests completed!")

if __name__ == "__main__":
    main()
