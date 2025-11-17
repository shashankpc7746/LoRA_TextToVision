"""
Test unified_video_generator integration with Scene Memory Core
Tests that Day 1 + Day 2 modules work together in production pipeline
"""

import pytest
import sys
from pathlib import Path

# Add AnimateDiff to path
sys.path.insert(0, str(Path(__file__).parent.parent / "AnimateDiff"))


def test_adaptive_engine_imports():
    """Test that all adaptive_engine modules can be imported"""
    from adaptive_engine import (
        get_story_context_parser,
        get_identity_memory,
        get_scene_memory,
        StoryContextParser,
        IdentityMemory,
        SceneMemoryCore
    )
    
    # Get singleton instances
    story_parser = get_story_context_parser()
    identity_memory = get_identity_memory()
    scene_memory = get_scene_memory()
    
    assert isinstance(story_parser, StoryContextParser)
    assert isinstance(identity_memory, IdentityMemory)
    assert isinstance(scene_memory, SceneMemoryCore)
    print("✅ All adaptive_engine modules imported successfully")


def test_day1_day2_integration():
    """Test that Day 1 story parser integrates with Day 2 scene memory"""
    from adaptive_engine import get_story_context_parser, get_scene_memory
    
    story_parser = get_story_context_parser()
    scene_memory = get_scene_memory()
    
    # Sample story
    sentences = [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests where ancient sages meditated",
        "The seeker meets a wise teacher at an old temple",
        "Together they discuss the nature of reality"
    ]
    
    # Day 1: Analyze story
    story_analysis = story_parser.analyze_story(sentences)
    
    assert len(story_analysis.characters) > 0
    assert 'seeker' in story_analysis.characters
    
    # Day 2: Build scene graph using Day 1 characters
    scene_graph = scene_memory.build_scene_graph(sentences, story_analysis.characters)
    
    assert scene_graph is not None
    
    # Verify scene graph contains entities
    stats = scene_memory.get_graph_stats()
    assert stats['total_scenes'] == 4
    assert stats['total_entities'] > 0
    
    # Verify entity tracking across scenes
    seeker_history = scene_memory.get_entity_history('seeker')
    assert seeker_history['total_appearances'] >= 2
    
    print(f"✅ Day 1 + Day 2 integration working:")
    print(f"   - Characters found: {list(story_analysis.characters.keys())}")
    print(f"   - Scenes: {stats['total_scenes']}")
    print(f"   - Entities: {stats['total_entities']}")
    print(f"   - Seeker appearances: {seeker_history['total_appearances']}")


def test_scene_memory_in_video_pipeline():
    """Test that scene memory provides useful context for video generation"""
    from adaptive_engine import get_story_context_parser, get_scene_memory
    
    story_parser = get_story_context_parser()
    scene_memory = get_scene_memory()
    
    sentences = [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests where ancient sages meditated",
        "The seeker meets a wise teacher at an old temple"
    ]
    
    # Analyze and build graph
    story_analysis = story_parser.analyze_story(sentences)
    scene_graph = scene_memory.build_scene_graph(sentences, story_analysis.characters)
    
    # Verify we can query scene data for each clip
    for i in range(len(sentences)):
        entities = scene_memory.get_entities_in_scene(i)
        assert entities is not None
        print(f"   Scene {i}: {entities}")
    
    # Verify we can track entity timeline
    seeker_timeline = scene_memory.get_entity_timeline('seeker')
    assert len(seeker_timeline) >= 2
    
    # Verify we can get transitions
    transitions = scene_memory.get_scene_transitions()
    assert len(transitions) == 2  # 3 scenes = 2 transitions
    
    print("✅ Scene memory provides useful video pipeline context")


def test_unified_video_generator_imports():
    """Test that unified_video_generator can import adaptive_engine modules"""
    # This simulates what happens in unified_video_generator.py
    try:
        from adaptive_engine import get_story_context_parser, get_identity_memory, get_scene_memory
        
        story_parser = get_story_context_parser()
        identity_memory = get_identity_memory()
        scene_memory = get_scene_memory()
        
        assert story_parser is not None
        assert identity_memory is not None
        assert scene_memory is not None
        
        print("✅ unified_video_generator can import all adaptive_engine modules")
        
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


if __name__ == "__main__":
    print("\n=== TESTING UNIFIED INTEGRATION ===\n")
    
    test_adaptive_engine_imports()
    print()
    
    test_day1_day2_integration()
    print()
    
    test_scene_memory_in_video_pipeline()
    print()
    
    test_unified_video_generator_imports()
    print()
    
    print("\n✅ ALL INTEGRATION TESTS PASSED")
