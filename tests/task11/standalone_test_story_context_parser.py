#!/usr/bin/env python3
"""
Standalone Test Suite for Story Context Parser
Runs without pytest for Day 1 validation
"""
import sys
import os
from pathlib import Path

# Setup paths
animatediff_path = Path(__file__).parent.parent.parent / "AnimateDiff"
sys.path.insert(0, str(animatediff_path))
os.chdir(str(animatediff_path))

from adaptive_engine.story_context_parser import (
    StoryContextParser,
    get_story_context_parser,
    Character,
    StoryAnalysis
)

def test_initialization():
    """Test parser initialization"""
    parser = StoryContextParser()
    assert parser is not None
    assert len(parser.male_indicators) > 0
    assert len(parser.female_indicators) > 0
    print("✅ test_initialization PASSED")

def test_gender_resolution_female():
    """Test gender resolution for female character"""
    parser = StoryContextParser()
    story = [
        "A young seeker begins her journey.",
        "She walks through the forest.",
        "The seeker finds her path."
    ]
    
    analysis = parser.analyze_story(story)
    
    # Find seeker character
    seeker = None
    for char_name, char in analysis.characters.items():
        if 'seeker' in char_name.lower():
            seeker = char
            break
    
    assert seeker is not None, "Seeker character not found"
    assert seeker.gender == 'female', f"Expected female, got {seeker.gender}"
    print(f"✅ test_gender_resolution_female PASSED (gender: {seeker.gender}, confidence: {seeker.confidence:.2f})")

def test_gender_consistency_across_sentences():
    """
    KEY TEST: Gender resolved correctly even when first sentence is ambiguous
    This solves the gender confusion problem!
    """
    parser = StoryContextParser()
    story = [
        "A young seeker begins the journey.",  # Ambiguous
        "She walks through misty forests.",    # Female
        "The seeker meets a wise teacher.",    # Ambiguous
        "She learns ancient wisdom."           # Female
    ]
    
    analysis = parser.analyze_story(story)
    
    # Find seeker
    seeker = None
    for char_name, char in analysis.characters.items():
        if 'seeker' in char_name.lower():
            seeker = char
            break
    
    assert seeker is not None, "Seeker character not found"
    assert seeker.gender == 'female', \
        f"Gender should be resolved as female from later sentences (LSTM-like analysis), got {seeker.gender}"
    print(f"✅ test_gender_consistency_across_sentences PASSED - Gender confusion SOLVED!")
    print(f"   Seeker identified as {seeker.gender} (confidence: {seeker.confidence:.2f})")

def test_multiple_characters():
    """Test handling multiple characters"""
    parser = StoryContextParser()
    story = [
        "A young student meets her teacher.",
        "He teaches her about meditation.",
        "She practices daily and grows stronger."
    ]
    
    analysis = parser.analyze_story(story)
    
    assert len(analysis.characters) >= 2, "Should identify at least 2 characters"
    print(f"✅ test_multiple_characters PASSED ({len(analysis.characters)} characters found)")

def test_enhanced_prompt_generation():
    """Test enhanced prompt generation"""
    parser = StoryContextParser()
    story = [
        "A young seeker begins her journey.",
        "She walks through the forest."
    ]
    
    analysis = parser.analyze_story(story)
    
    assert len(analysis.enhanced_prompts) == len(story)
    # Check that prompts were enhanced
    enhanced = any(analysis.enhanced_prompts[i] != story[i] for i in range(len(story)))
    assert enhanced, "Prompts should be enhanced"
    print(f"✅ test_enhanced_prompt_generation PASSED")
    print(f"   Original: {story[0][:50]}...")
    print(f"   Enhanced: {analysis.enhanced_prompts[0][:50]}...")

def test_empty_story():
    """Test handling empty story"""
    parser = StoryContextParser()
    story = []
    
    analysis = parser.analyze_story(story)
    
    assert analysis.total_sentences == 0
    assert len(analysis.characters) == 0
    print("✅ test_empty_story PASSED")

def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("🧪 Running Story Context Parser Tests - Task 11 Day 1")
    print("=" * 70)
    print()
    
    tests = [
        test_initialization,
        test_gender_resolution_female,
        test_gender_consistency_across_sentences,
        test_multiple_characters,
        test_enhanced_prompt_generation,
        test_empty_story
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
