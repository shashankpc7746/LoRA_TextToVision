#!/usr/bin/env python3
"""
Test Style Improvements Script
Generate videos for all three styles (realistic, anime, artistic) to validate improvements
"""

import os
import sys
import time
import json
from datetime import datetime

def create_test_lesson():
    """Create a test lesson for style validation"""
    test_lesson = {
        "title": "Style Improvement Test - Character Consistency and Background Clarity",
        "level": "Test",
        "text": "A beautiful young woman walks through a peaceful mountain landscape. She pauses to admire the majestic Himalayan peaks in the distance. The woman continues her journey through the serene forest path, surrounded by tall ancient trees. She reaches a sacred temple where she sits in meditation, finding inner peace in the tranquil spiritual setting.",
        "scenes": [
            {
                "description": "A beautiful young woman walks through a peaceful mountain landscape",
                "duration": 4.0
            },
            {
                "description": "She pauses to admire the majestic Himalayan peaks in the distance",
                "duration": 4.0
            },
            {
                "description": "The woman continues her journey through the serene forest path, surrounded by tall ancient trees",
                "duration": 4.0
            },
            {
                "description": "She reaches a sacred temple where she sits in meditation, finding inner peace in the tranquil spiritual setting",
                "duration": 4.0
            }
        ],
        "tts": True
    }
    
    # Save test lesson
    lesson_path = "lessons/test_style_improvements.json"
    with open(lesson_path, 'w', encoding='utf-8') as f:
        json.dump(test_lesson, f, indent=2, ensure_ascii=False)
    
    return lesson_path

def generate_style_test_videos():
    """Generate test videos for all three styles"""
    
    print("🎬 STYLE IMPROVEMENT VALIDATION TEST")
    print("=" * 60)
    
    # Create test lesson
    lesson_path = create_test_lesson()
    print(f"✅ Created test lesson: {lesson_path}")
    
    styles = ["realistic", "anime", "artistic"]
    results = {}
    
    for style in styles:
        print(f"\n🎯 TESTING STYLE: {style.upper()}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Import the unified video generator
            from unified_video_generator import UnifiedVideoGenerator
            
            # Generate video for this style
            generator = UnifiedVideoGenerator()
            result = generator.generate_complete_video(lesson_path, style, speech_rate=1)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if result:
                print(f"✅ {style.upper()} video generated successfully!")
                print(f"📁 Output: {result}")
                print(f"⏱️ Generation time: {generation_time:.1f} seconds")
                
                results[style] = {
                    "success": True,
                    "output_path": result,
                    "generation_time": generation_time
                }
            else:
                print(f"❌ {style.upper()} video generation failed!")
                results[style] = {
                    "success": False,
                    "error": "Generation returned None",
                    "generation_time": generation_time
                }
            
            generator.cleanup()
            
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            
            print(f"❌ {style.upper()} video generation failed with error: {e}")
            results[style] = {
                "success": False,
                "error": str(e),
                "generation_time": generation_time
            }
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 STYLE IMPROVEMENT TEST SUMMARY")
    print("=" * 60)
    
    for style, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        time_str = f"{result['generation_time']:.1f}s"
        
        print(f"{style.upper():10} | {status:10} | {time_str:8}")
        
        if result["success"]:
            print(f"           | Output: {result['output_path']}")
        else:
            print(f"           | Error: {result.get('error', 'Unknown error')}")
    
    print("\n🎬 Test completed! Check the generated videos for:")
    print("   - REALISTIC: Character face consistency, clear backgrounds, smooth motion")
    print("   - ANIME: Vibrant colors, clear backgrounds, character consistency")
    print("   - ARTISTIC: Clear backgrounds, prompt understanding, character consistency")
    
    return results

if __name__ == "__main__":
    # Change to AnimateDiff directory if not already there
    if not os.path.exists("lessons"):
        print("❌ Error: Please run this script from the AnimateDiff directory!")
        sys.exit(1)
    
    # Run the style improvement tests
    results = generate_style_test_videos()
    
    # Exit with appropriate code
    all_success = all(result["success"] for result in results.values())
    sys.exit(0 if all_success else 1)
