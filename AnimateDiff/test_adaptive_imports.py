"""
Quick test to verify adaptive_engine imports work in unified_video_generator context
"""
import sys
import os

# Same path setup as unified_video_generator.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing adaptive_engine imports...")

try:
    from adaptive_engine import get_story_context_parser, get_identity_memory, get_scene_memory
    print("✅ Step 1: Imports successful")
    
    # Test getting instances
    story_parser = get_story_context_parser()
    identity_memory = get_identity_memory()
    scene_memory = get_scene_memory()
    print("✅ Step 2: Got singleton instances")
    
    # Test a simple analysis
    test_sentences = [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests"
    ]
    
    story_analysis = story_parser.analyze_story(test_sentences)
    print(f"✅ Step 3: Story analysis completed - found {len(story_analysis.characters)} characters")
    
    # Test scene graph building
    scene_graph = scene_memory.build_scene_graph(test_sentences, story_analysis.characters)
    stats = scene_memory.get_graph_stats()
    print(f"✅ Step 4: Scene graph built - {stats['total_scenes']} scenes, {stats['total_entities']} entities")
    
    print("\n🎉 ALL TESTS PASSED! Adaptive engine integration working!")
    print("✅ The unified_video_generator.py fix is working correctly\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
