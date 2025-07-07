#!/usr/bin/env python3
"""
Enhanced Prompt Processing System
Converts video-optimized prompts into story-format audio prompts
"""

import re
import os
import sys
from typing import List, Dict, Tuple
import google.generativeai as genai
from dataclasses import dataclass

# Add tts_module to path for translation_agent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tts_module'))

@dataclass
class EnhancedPrompt:
    """Container for enhanced prompt data"""
    original: str
    video_prompt: str  # Optimized for video generation
    audio_prompt: str  # Enhanced for story narration
    has_dialogue: bool  # Whether this contains character speech
    character_gender: str  # 'male', 'female', or 'neutral'
    dialogue_text: str = ""  # Extracted dialogue if any

class PromptEnhancer:
    """Enhanced prompt processing for audio-video integration"""
    
    def __init__(self, api_key: str = None):
        """Initialize the prompt enhancer with Gemini API"""
        if api_key:
            genai.configure(api_key=api_key)
        else:
            # Try to get API key from environment or config
            try:
                from translation_agent import genai
                print("✅ Using existing Gemini API configuration")
            except ImportError:
                print("⚠️ Warning: No Gemini API key configured. Using fallback enhancement.")
        
        self.model = None
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini model initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Gemini model: {e}")
    
    def detect_character_info(self, text: str) -> Tuple[bool, str, str]:
        """
        Detect if text contains dialogue and character gender
        Returns: (has_dialogue, gender, dialogue_text)
        """
        # Look for dialogue patterns
        dialogue_patterns = [
            r'"([^"]*)"',  # "spoken text"
            r"'([^']*)'",  # 'spoken text'
            r'he says?[:\s]*"([^"]*)"',  # he says: "text"
            r'she says?[:\s]*"([^"]*)"',  # she says: "text"
            r'thinks?[:\s]*"([^"]*)"',  # thinks: "text"
        ]
        
        dialogue_text = ""
        has_dialogue = False
        
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                has_dialogue = True
                dialogue_text = matches[0]  # Take first match
                break
        
        # Detect gender from text
        gender = "neutral"
        male_indicators = ["boy", "man", "he", "his", "him", "male", "guy", "father", "brother"]
        female_indicators = ["girl", "woman", "she", "her", "female", "lady", "mother", "sister"]
        
        text_lower = text.lower()
        male_count = sum(1 for word in male_indicators if word in text_lower)
        female_count = sum(1 for word in female_indicators if word in text_lower)
        
        if male_count > female_count:
            gender = "male"
        elif female_count > male_count:
            gender = "female"
        
        return has_dialogue, gender, dialogue_text
    
    def enhance_prompt_with_ai(self, original_prompt: str) -> str:
        """Use Gemini AI to enhance video prompt for audio narration"""
        if not self.model:
            return self.fallback_enhancement(original_prompt)
        
        try:
            enhancement_prompt = f"""
Transform this video generation prompt into engaging story narration suitable for audio:

Original prompt: "{original_prompt}"

Requirements:
1. Convert technical video descriptions into flowing narrative prose
2. Add emotional depth and storytelling elements
3. Make it sound natural when spoken aloud
4. Keep the core visual elements but make them story-like
5. Add atmospheric details that enhance the mood
6. If there's character dialogue, preserve it clearly

Return only the enhanced narrative text, nothing else.
"""
            
            response = self.model.generate_content(enhancement_prompt)
            enhanced = response.text.strip()
            
            # Clean up any markdown or formatting
            enhanced = re.sub(r'[*_`]', '', enhanced)
            enhanced = enhanced.replace('\n', ' ').strip()
            
            return enhanced
            
        except Exception as e:
            print(f"⚠️ AI enhancement failed: {e}. Using fallback.")
            return self.fallback_enhancement(original_prompt)
    
    def fallback_enhancement(self, original_prompt: str) -> str:
        """Fallback enhancement when AI is not available"""
        # Simple rule-based enhancement
        enhanced = original_prompt
        
        # Add narrative flow
        if enhanced.startswith("Anime boy"):
            enhanced = enhanced.replace("Anime boy", "A young man")
        elif enhanced.startswith("Anime girl"):
            enhanced = enhanced.replace("Anime girl", "A young woman")
        
        # Add atmospheric words
        replacements = {
            "walks": "strolls quietly",
            "stops": "pauses thoughtfully",
            "looks": "gazes",
            "runs": "hurries",
            "stands": "stands silently",
            "sits": "settles down",
        }
        
        for old, new in replacements.items():
            enhanced = enhanced.replace(old, new)
        
        # Add connecting words for flow
        if not enhanced.endswith('.'):
            enhanced += '.'
        
        return enhanced
    
    def process_prompt(self, original_prompt: str) -> EnhancedPrompt:
        """Process a single prompt and return enhanced version"""
        print(f"🔄 Processing: {original_prompt[:50]}...")
        
        # Detect character information
        has_dialogue, gender, dialogue_text = self.detect_character_info(original_prompt)
        
        # Keep original as video prompt (it's already optimized for video)
        video_prompt = original_prompt
        
        # Enhance for audio narration
        audio_prompt = self.enhance_prompt_with_ai(original_prompt)
        
        enhanced = EnhancedPrompt(
            original=original_prompt,
            video_prompt=video_prompt,
            audio_prompt=audio_prompt,
            has_dialogue=has_dialogue,
            character_gender=gender,
            dialogue_text=dialogue_text
        )
        
        print(f"✅ Enhanced: {audio_prompt[:50]}...")
        if has_dialogue:
            print(f"🗣️ Dialogue detected ({gender}): {dialogue_text}")
        
        return enhanced
    
    def process_prompt_list(self, prompts: List[str]) -> List[EnhancedPrompt]:
        """Process a list of prompts"""
        print(f"🎬 Processing {len(prompts)} prompts for enhancement...")
        
        enhanced_prompts = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt {i}/{len(prompts)}:")
            enhanced = self.process_prompt(prompt)
            enhanced_prompts.append(enhanced)
        
        print(f"\n✅ Successfully enhanced {len(enhanced_prompts)} prompts!")
        return enhanced_prompts

def test_prompt_enhancer():
    """Test the prompt enhancer with sample data"""
    print("🧪 Testing Prompt Enhancer...")
    
    # Sample prompts from multi_clip_generator.py
    test_prompts = [
        "Anime boy wearing a hoodie walks on a quiet street under a grey sky.",
        "Rain falls gently on anime boy as soft wind moves the hoodie.",
        "Anime boy stops at a glowing vending machine beside the road.",
        "Anime boy buys a warm canned coffee and holds the coffee with both hands.",
        "A small dog runs past anime boy, splashing water in anime style."
    ]
    
    enhancer = PromptEnhancer()
    enhanced_prompts = enhancer.process_prompt_list(test_prompts)
    
    print(f"\n{'='*60}")
    print("📊 ENHANCEMENT RESULTS:")
    print(f"{'='*60}")
    
    for i, enhanced in enumerate(enhanced_prompts, 1):
        print(f"\n🎬 PROMPT {i}:")
        print(f"Original: {enhanced.original}")
        print(f"Audio:    {enhanced.audio_prompt}")
        print(f"Gender:   {enhanced.character_gender}")
        print(f"Dialogue: {enhanced.has_dialogue}")

if __name__ == "__main__":
    test_prompt_enhancer()
