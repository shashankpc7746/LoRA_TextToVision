#!/usr/bin/env python3
"""
Test script to verify prompt splitting works correctly
"""

import re

def split_paragraph(text):
    """Split paragraph by lines and sentence punctuation."""
    # First try splitting by newlines (for line-based prompts)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    # If we get multiple lines, use them
    if len(lines) > 1:
        return lines
    
    # Otherwise, fall back to sentence splitting
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# Test cases
test_cases = [
    # Case 1: Line-based prompts (your current format)
    """
Anime boy wearing a hoodie walks on a quiet street under a grey sky.
Rain falls gently on anime boy as soft wind moves the hoodie.
Anime boy stops at a glowing vending machine beside the road.
Anime boy buys a warm canned coffee and holds the coffee with both hands.
A small dog runs past anime boy, splashing water in anime style.
Anime boy smiles and starts walking again through the calm street.
Anime boy passes an anime bakery with warm yellow lights in the window.
Anime boy pauses and looks inside the bakery as steam fogs the glass.
Anime boy stands near a train crossing while red lights start flashing.
Anime boy drinks the coffee slowly as the train moves fast through the rain.
""",
    
    # Case 2: Sentence-based prompts
    "A boy walks through a desert as sand swirls around him. He shields his eyes from the blazing sun and then picks up a strange glowing artifact. Suddenly, the sky turns purple and lightning flashes in the distance. The boy starts to run, clutching the artifact tightly.",
    
    # Case 3: Mixed format
    """
The girl walks down a quiet city street at sunrise.
The girl looks up at the tall buildings glowing in the morning light.
The girl pulls out her camera and takes a photo. The girl hears music nearby and follows the sound.
The girl smiles as she finds a street artist playing guitar on the corner.
"""
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*50}")
    print(f"TEST CASE {i}:")
    print(f"{'='*50}")
    
    prompts = split_paragraph(test_case)
    print(f"🧠 Detected {len(prompts)} prompts:")
    
    for j, prompt in enumerate(prompts, 1):
        print(f"   [{j}] Length: {len(prompt)} chars - {prompt}")
    
    if len(prompts) <= 1:
        print("⚠️ Warning: Only 1 prompt detected!")
    else:
        print(f"✅ Successfully split into {len(prompts)} prompts!")

print(f"\n{'='*50}")
print("✅ Prompt splitting test completed!")
print(f"{'='*50}")
