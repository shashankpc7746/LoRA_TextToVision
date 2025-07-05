#!/usr/bin/env python3

import requests
import json
import time

# Test the translation functionality
API_URL = "http://192.168.0.121:8001/api/generate-and-sync"

def test_translation(text, target_lang, lang_name):
    """Test translation for a specific language"""
    print(f"\n🌍 Testing {lang_name} ({target_lang}) translation...")
    print(f"Original text: '{text}'")
    
    try:
        # Make the API request
        data = {
            'text': text,
            'target_lang': target_lang
        }
        
        print(f"Making request to: {API_URL}")
        print(f"Data: {data}")
        
        response = requests.post(API_URL, data=data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ SUCCESS!")
            print(f"Session ID: {result.get('session_id', 'N/A')}")
            print(f"Video URL: {result.get('video_url', 'N/A')}")
            print(f"Audio URL: {result.get('audio_url', 'N/A')}")
            print(f"Original Text: {result.get('original_text', 'N/A')}")
            print(f"Translated Text: {result.get('translated_text', 'N/A')}")
            print(f"Translation Confidence: {result.get('translation_confidence', 'N/A')}")
            return True
        else:
            print(f"❌ FAILED! Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    print("🚀 Testing Translation Functionality")
    print("=" * 50)
    
    # Test text
    test_text = "Hello, how are you today?"
    
    # Test different languages
    languages_to_test = [
        ('es', 'Spanish'),
        ('fr', 'French'),
        ('hi', 'Hindi'),
        ('de', 'German'),
        ('zh', 'Chinese')
    ]
    
    results = []
    
    for lang_code, lang_name in languages_to_test:
        success = test_translation(test_text, lang_code, lang_name)
        results.append((lang_name, success))
        
        # Wait between requests to avoid overwhelming the server
        if lang_code != languages_to_test[-1][0]:  # Don't wait after the last request
            print("⏳ Waiting 10 seconds before next test...")
            time.sleep(10)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TRANSLATION TEST SUMMARY")
    print("=" * 50)
    
    successful = 0
    for lang_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{lang_name}: {status}")
        if success:
            successful += 1
    
    print(f"\nTotal: {successful}/{len(results)} languages working")
    
    if successful == len(results):
        print("🎉 ALL TRANSLATION TESTS PASSED!")
    else:
        print("⚠️  Some translation tests failed.")

if __name__ == "__main__":
    main()
