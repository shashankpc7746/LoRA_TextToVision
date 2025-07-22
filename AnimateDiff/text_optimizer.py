#!/usr/bin/env python3
"""
Text Optimizer using Gemini API
Converts lesson text into optimized video prompts and audio/subtitle content
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextOptimizer:
    def __init__(self):
        self.gemini_api_key = os.getenv('GOOGLE_GEMINI_API_KEY')
        self.gemini_url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
        
    def optimize_lesson_text(self, lesson_text, lesson_title="Lesson"):
        """
        Optimize lesson text into video prompts and audio/subtitle content
        """
        prompt = f"""
You are an expert content optimizer for text-to-video generation. Convert the following lesson text into two optimized parts:

**LESSON TITLE:** {lesson_title}

**ORIGINAL TEXT:**
{lesson_text}

**TASK:** Create exactly 6-8 scenes. Each scene should be 15-20 seconds long.

**OUTPUT FORMAT (JSON):**
{{
  "video_prompts": [
    "Scene 1: [Short, visual description with clear character, setting, and action]",
    "Scene 2: [Next scene with character continuity and clear visuals]",
    ...
  ],
  "audio_script": [
    "Scene 1: [Concise narration that matches the video, 15-20 seconds of speech]",
    "Scene 2: [Next narration segment, continuing the story]",
    ...
  ]
}}

**REQUIREMENTS:**

**For Video Prompts:**
- Each prompt should be 10-15 words maximum
- Include specific character descriptions (age, gender, clothing, appearance)
- Mention clear visual elements (setting, objects, actions)
- Maintain character consistency across scenes
- Use cinematic language (close-up, wide shot, etc.)
- Example: "Young teacher in white robes explains ancient scrolls in temple courtyard"

**For Audio Script:**
- Each segment should be 20-30 words (15-20 seconds of speech)
- Tell a coherent story that matches the video prompts
- Use simple, clear language
- Maintain narrative flow between segments
- Focus on the key educational content
- DO NOT include "Scene 1:", "Scene 2:" etc. - only the story content
- Example: "Welcome to the ancient world of Vedic wisdom. These sacred texts hold secrets passed down through generations."

**IMPORTANT:**
- Video prompts and audio script must have the same number of scenes
- Each audio segment should match its corresponding video prompt
- Keep the educational essence of the original content
- Make it engaging and story-driven
- Ensure smooth transitions between scenes

Generate the optimized content now:
"""

        try:
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 2048
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(self.gemini_url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from the response
                try:
                    # Find JSON content between ```json and ```
                    if "```json" in generated_text:
                        json_start = generated_text.find("```json") + 7
                        json_end = generated_text.find("```", json_start)
                        json_content = generated_text[json_start:json_end].strip()
                    else:
                        # Try to find JSON content directly
                        json_start = generated_text.find("{")
                        json_end = generated_text.rfind("}") + 1
                        json_content = generated_text[json_start:json_end]
                    
                    optimized_content = json.loads(json_content)
                    
                    # Validate the structure
                    if "video_prompts" in optimized_content and "audio_script" in optimized_content:
                        if len(optimized_content["video_prompts"]) == len(optimized_content["audio_script"]):
                            return optimized_content
                        else:
                            print("Warning: Video prompts and audio script have different lengths")
                            return optimized_content
                    else:
                        print("Error: Invalid response structure from Gemini")
                        return None
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON from Gemini response: {e}")
                    print(f"Raw response: {generated_text}")
                    return None
                    
            else:
                print(f"Error calling Gemini API: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error in text optimization: {e}")
            return None
    
    def create_optimized_lesson(self, original_lesson_path, output_lesson_path):
        """
        Create an optimized lesson file from the original lesson
        """
        try:
            # Load original lesson
            with open(original_lesson_path, 'r', encoding='utf-8') as f:
                lesson_data = json.load(f)
            
            original_text = lesson_data.get('text', '')
            lesson_title = lesson_data.get('title', 'Lesson')
            
            print(f"Optimizing lesson: {lesson_title}")
            print(f"Original text length: {len(original_text)} characters")
            
            # Optimize the text
            optimized_content = self.optimize_lesson_text(original_text, lesson_title)
            
            if optimized_content:
                # Create new lesson structure
                optimized_lesson = {
                    "title": lesson_title,
                    "level": lesson_data.get('level', 'optimized'),
                    "text": " ".join(optimized_content["audio_script"]),  # Combined audio script
                    "video_prompts": optimized_content["video_prompts"],
                    "audio_script": optimized_content["audio_script"],
                    "scenes": [],
                    "tts": True,
                    "optimized": True
                }
                
                # Create scenes from optimized content
                for i, (video_prompt, audio_text) in enumerate(zip(
                    optimized_content["video_prompts"], 
                    optimized_content["audio_script"]
                )):
                    scene = {
                        "description": video_prompt,
                        "audio_text": audio_text,
                        "duration": 4.0  # 15-20 seconds / 4 = ~4 seconds per scene
                    }
                    optimized_lesson["scenes"].append(scene)
                
                # Save optimized lesson
                with open(output_lesson_path, 'w', encoding='utf-8') as f:
                    json.dump(optimized_lesson, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Optimized lesson created: {output_lesson_path}")
                print(f"📊 Generated {len(optimized_content['video_prompts'])} scenes")
                print(f"🎬 Video prompts: {len(optimized_content['video_prompts'])}")
                print(f"🎵 Audio segments: {len(optimized_content['audio_script'])}")
                
                return output_lesson_path
            else:
                print("❌ Failed to optimize lesson text")
                return None
                
        except Exception as e:
            print(f"❌ Error creating optimized lesson: {e}")
            return None

def main():
    """Test the text optimizer"""
    optimizer = TextOptimizer()
    
    # Test with the API lesson file
    original_path = "lessons/api_lesson_20250722_152840.json"
    output_path = "lessons/api_lesson_20250722_152840_optimized.json"
    
    if os.path.exists(original_path):
        result = optimizer.create_optimized_lesson(original_path, output_path)
        if result:
            print(f"\n🎯 SUCCESS! Optimized lesson ready for video generation:")
            print(f"📁 File: {result}")
        else:
            print("\n❌ FAILED! Could not optimize lesson")
    else:
        print(f"❌ Original lesson file not found: {original_path}")

if __name__ == "__main__":
    main()
