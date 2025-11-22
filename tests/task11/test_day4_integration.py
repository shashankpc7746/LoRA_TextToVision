"""
Test Day 3 & Day 4 integration into unified_video_generator.py
"""

import sys
import os

# Add AnimateDiff to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all Days 1-4 modules can be imported together"""
    print("\n🧪 Testing Days 1-4 module imports...")
    
    try:
        from adaptive_engine import (
            get_story_context_parser,
            get_identity_memory,
            get_scene_memory,
            get_narrative_sequencer,
            get_emotion_controller
        )
        print("   ✅ All Days 1-4 modules imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_full_pipeline():
    """Test complete Days 1-4 analysis pipeline"""
    print("\n🧪 Testing complete Days 1-4 pipeline...")
    
    from adaptive_engine import (
        get_story_context_parser,
        get_scene_memory,
        get_narrative_sequencer,
        get_emotion_controller
    )
    
    # Test story
    sentences = [
        "A young spiritual seeker begins her journey",
        "She walks through the mystical forest",
        "The seeker meets a wise teacher in the temple",
        "They discuss the nature of reality",
        "She realizes the profound truth"
    ]
    
    # Day 1: Story analysis
    print("\n   Day 1: Story Context Analysis...")
    story_parser = get_story_context_parser()
    story_analysis = story_parser.analyze_story(sentences)
    print(f"      ✅ Found {len(story_analysis.characters)} characters")
    
    # Day 2: Scene graph
    print("\n   Day 2: Scene Memory Graph...")
    scene_memory = get_scene_memory()
    scene_graph = scene_memory.build_scene_graph(sentences, story_analysis.characters)
    stats = scene_memory.get_graph_stats()
    print(f"      ✅ Scene graph: {stats['total_scenes']} scenes, {stats['total_entities']} entities")
    
    # Day 3: Narrative analysis
    print("\n   Day 3: Narrative Sequencing...")
    narrative_sequencer = get_narrative_sequencer()
    narrative_analysis = narrative_sequencer.analyze_narrative(sentences, story_analysis.characters)
    print(f"      ✅ Story structure: {len(narrative_analysis.story_beats)} beats, {len(narrative_analysis.character_arcs)} arcs")
    
    # Day 4: Emotion tracking
    print("\n   Day 4: Emotion Controller...")
    emotion_controller = get_emotion_controller()
    
    # Set emotions based on story beats
    emotion_mapping = {
        'SETUP': 'neutral',
        'RISING_ACTION': 'fear',
        'CLIMAX': 'surprise',
        'FALLING_ACTION': 'sadness',
        'RESOLUTION': 'joy',
        'TWIST': 'surprise'
    }
    
    for char_name in story_analysis.characters.keys():
        for i, beat in enumerate(narrative_analysis.story_beats):
            emotion = emotion_mapping.get(beat.beat_type.name, 'neutral')
            intensity = beat.tension_level
            emotion_controller.set_emotion(char_name, emotion, intensity, scene_index=i)
    
    print(f"      ✅ Emotions set for {len(story_analysis.characters)} characters")
    
    # Verify emotion-motion coupling for last scene
    last_scene_idx = len(sentences) - 1
    for char_name in story_analysis.characters.keys():
        emotion_state = emotion_controller.get_current_emotion(char_name, last_scene_idx)
        if emotion_state:
            base_intensity = 1.0
            motion_intensity = emotion_controller.get_motion_intensity(emotion_state.emotion, base_intensity)
            print(f"      • {char_name} (scene {last_scene_idx}): {emotion_state.emotion} → motion {motion_intensity:.2f}x")
    
    print("\n   ✅ Complete Days 1-4 pipeline working!")
    return True

def test_production_integration_pattern():
    """Test the exact pattern used in unified_video_generator.py"""
    print("\n🧪 Testing production integration pattern...")
    
    from adaptive_engine import (
        get_story_context_parser,
        get_identity_memory,
        get_scene_memory,
        get_narrative_sequencer,
        get_emotion_controller
    )
    
    # Simulate the exact code from unified_video_generator.py
    sentences = ["A seeker begins journey", "She walks forest", "The seeker meets teacher"]
    
    story_parser = get_story_context_parser()
    identity_memory = get_identity_memory()
    scene_memory = get_scene_memory()
    narrative_sequencer = get_narrative_sequencer()
    emotion_controller = get_emotion_controller()
    
    # Day 1
    story_analysis = story_parser.analyze_story(sentences)
    
    # Day 2
    scene_graph = scene_memory.build_scene_graph(sentences, story_analysis.characters)
    graph_stats = scene_memory.get_graph_stats()
    
    # Day 3
    narrative_analysis = narrative_sequencer.analyze_narrative(sentences, story_analysis.characters)
    
    # Day 4: Emotion initialization (exact pattern from unified_video_generator.py)
    for char_name in story_analysis.characters.keys():
        for i, beat in enumerate(narrative_analysis.story_beats):
            emotion_mapping = {
                'SETUP': 'neutral',
                'RISING_ACTION': 'fear',
                'CLIMAX': 'surprise',
                'FALLING_ACTION': 'sadness',
                'RESOLUTION': 'joy',
                'TWIST': 'surprise'
            }
            emotion = emotion_mapping.get(beat.beat_type.name, 'neutral')
            intensity = beat.tension_level
            emotion_controller.set_emotion(char_name, emotion, intensity, scene_index=i)
    
    print(f"   ✅ Production pattern works: {len(story_analysis.characters)} chars, {graph_stats['total_scenes']} scenes")
    print(f"   ✅ Narrative: {len(narrative_analysis.story_beats)} beats")
    print(f"   ✅ Emotions: Initialized for all characters")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TESTING DAYS 3-4 PRODUCTION INTEGRATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Full pipeline
    if not test_full_pipeline():
        all_passed = False
    
    # Test 3: Production pattern
    if not test_production_integration_pattern():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✅ ALL INTEGRATION TESTS PASSED!")
        print("  ✅ Days 3-4 ready for production video generation")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("="*60 + "\n")
