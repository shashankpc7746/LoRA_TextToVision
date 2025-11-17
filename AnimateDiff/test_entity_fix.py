"""
Test the specific fix for get_entities_in_scene() returning dicts
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_engine import get_scene_memory, get_story_context_parser

print("Testing entity extraction fix...\n")

# Create test data
story_parser = get_story_context_parser()
scene_memory = get_scene_memory()

test_sentences = [
    "A young seeker embarks on a spiritual journey",
    "She walks through misty forests where ancient sages meditated",
    "The seeker meets a wise teacher at an old temple"
]

# Analyze and build scene graph
story_analysis = story_parser.analyze_story(test_sentences)
scene_graph = scene_memory.build_scene_graph(test_sentences, story_analysis.characters)

print("Testing get_entities_in_scene()...\n")

# Test each scene
for i in range(len(test_sentences)):
    entities_in_scene = scene_memory.get_entities_in_scene(i)
    
    print(f"Scene {i}: {test_sentences[i][:50]}...")
    print(f"  Raw return type: {type(entities_in_scene)}")
    print(f"  Number of entities: {len(entities_in_scene)}")
    
    if entities_in_scene:
        print(f"  First entity type: {type(entities_in_scene[0])}")
        print(f"  First entity: {entities_in_scene[0]}")
        
        # Test the fix: extract names
        entity_names = [e['name'] for e in entities_in_scene]
        print(f"  ✅ Extracted names: {', '.join(entity_names[:3])}")
    
    print()

print("✅ Fix verified! Entity names can be extracted and joined as strings.")
