#!/usr/bin/env python3
"""
Test Individual Styles Script
Generate videos for anime and artistic styles individually
"""

import os
import sys
import time
import subprocess

def test_anime_style():
    """Test anime style generation"""
    print("🎯 TESTING ANIME STYLE")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # Generate anime video
        result = subprocess.run([
            sys.executable, "generate_lesson_video_safe.py", 
            "test_style_improvements.json", "anime", "1"
        ], capture_output=True, text=True, timeout=900)  # 15 minute timeout
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ ANIME video generated successfully!")
            print(f"⏱️ Generation time: {generation_time:.1f} seconds")
            return True
        else:
            print(f"❌ ANIME video generation failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ ANIME video generation timed out after 15 minutes")
        return False
    except Exception as e:
        print(f"❌ ANIME video generation failed with error: {e}")
        return False

def test_artistic_style():
    """Test artistic style generation"""
    print("\n🎯 TESTING ARTISTIC STYLE")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        # Generate artistic video
        result = subprocess.run([
            sys.executable, "generate_lesson_video_safe.py", 
            "test_style_improvements.json", "artistic", "1"
        ], capture_output=True, text=True, timeout=900)  # 15 minute timeout
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ ARTISTIC video generated successfully!")
            print(f"⏱️ Generation time: {generation_time:.1f} seconds")
            return True
        else:
            print(f"❌ ARTISTIC video generation failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ ARTISTIC video generation timed out after 15 minutes")
        return False
    except Exception as e:
        print(f"❌ ARTISTIC video generation failed with error: {e}")
        return False

def main():
    """Main test function"""
    print("🎬 INDIVIDUAL STYLE TESTING")
    print("=" * 60)
    
    # Change to correct directory if needed
    if not os.path.exists("lessons"):
        print("❌ Error: Please run this script from the AnimateDiff directory!")
        sys.exit(1)
    
    results = {}
    
    # Test anime style
    results["anime"] = test_anime_style()
    
    # Test artistic style  
    results["artistic"] = test_artistic_style()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 INDIVIDUAL STYLE TEST SUMMARY")
    print("=" * 60)
    
    for style, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{style.upper():10} | {status}")
    
    print("\n🎬 Test completed!")
    print("Check the generated videos for:")
    print("   - ANIME: Vibrant colors, clear backgrounds, character consistency")
    print("   - ARTISTIC: Clear backgrounds, prompt understanding, character consistency")
    
    # Exit with appropriate code
    all_success = all(results.values())
    sys.exit(0 if all_success else 1)

if __name__ == "__main__":
    main()
