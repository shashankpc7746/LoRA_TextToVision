"""
Unit Tests for Scene Memory Core - Day 2 of Task 11
Tests scene graph construction, entity tracking, and query API
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "AnimateDiff"))

from AnimateDiff.adaptive_engine.scene_memory_core import (
    SceneMemoryCore,
    SceneNode,
    EntityNode,
    get_scene_memory
)


@pytest.fixture
def sample_story():
    """Sample story for testing"""
    return [
        "A young seeker begins her journey through the misty forest.",
        "She encounters a wise teacher at the mountain temple.",
        "The teacher shares ancient wisdom with the seeker.",
        "Together they walk the sacred path through the valley."
    ]


@pytest.fixture
def scene_memory():
    """Fresh scene memory instance"""
    return SceneMemoryCore(cache_dir="cache/test_scene_memory")


def test_scene_memory_initialization(scene_memory):
    """Test that scene memory initializes correctly"""
    assert scene_memory is not None
    assert scene_memory.graph is not None
    assert len(scene_memory.scenes) == 0
    assert len(scene_memory.entities) == 0
    print("✅ Scene memory initialized")


def test_build_scene_graph(scene_memory, sample_story):
    """Test scene graph construction"""
    graph = scene_memory.build_scene_graph(sample_story)
    
    assert graph is not None
    assert len(scene_memory.scenes) == 4  # 4 sentences = 4 scenes
    assert len(scene_memory.entities) > 0  # Should detect entities
    
    # Check scene nodes exist
    for i in range(4):
        scene_id = f"scene_{i}"
        assert scene_id in scene_memory.scenes
        assert scene_memory.graph.has_node(scene_id)
    
    print(f"✅ Scene graph built: {len(scene_memory.scenes)} scenes, {len(scene_memory.entities)} entities")


def test_entity_extraction(scene_memory, sample_story):
    """Test entity extraction from sentences"""
    scene_memory.build_scene_graph(sample_story)
    
    # Should detect common entities
    entity_names = [e.name for e in scene_memory.entities.values()]
    
    # Check for known entities
    assert 'seeker' in entity_names or any('seeker' in name for name in entity_names)
    assert 'teacher' in entity_names or any('teacher' in name for name in entity_names)
    
    print(f"✅ Entities extracted: {entity_names}")


def test_entity_tracking(scene_memory, sample_story):
    """Test entity appearance tracking"""
    scene_memory.build_scene_graph(sample_story)
    
    # Find seeker entity
    seeker_entity = None
    for entity in scene_memory.entities.values():
        if 'seeker' in entity.name.lower():
            seeker_entity = entity
            break
    
    assert seeker_entity is not None
    assert seeker_entity.total_appearances >= 2  # Appears in multiple scenes
    assert seeker_entity.first_appearance >= 0
    assert seeker_entity.last_appearance >= seeker_entity.first_appearance
    
    print(f"✅ Seeker tracked: {seeker_entity.total_appearances} appearances")


def test_scene_transitions(scene_memory, sample_story):
    """Test scene transition edges"""
    scene_memory.build_scene_graph(sample_story)
    
    transitions = scene_memory.get_scene_transitions()
    
    # Should have 3 transitions (scene 0→1, 1→2, 2→3)
    assert len(transitions) == 3
    
    # Check first transition
    assert transitions[0]['from_scene'] == 'scene_0'
    assert transitions[0]['to_scene'] == 'scene_1'
    assert transitions[0]['transition_type'] == 'temporal_next'
    
    print(f"✅ {len(transitions)} transitions detected")


def test_get_entity_history(scene_memory, sample_story):
    """Test entity history query"""
    scene_memory.build_scene_graph(sample_story)
    
    history = scene_memory.get_entity_history('seeker')
    
    assert history['found'] == True
    assert 'seeker' in history['entity_name'].lower()
    assert history['total_appearances'] >= 1
    assert 'scenes' in history
    assert len(history['scenes']) > 0
    
    print(f"✅ Entity history: seeker appeared {history['total_appearances']} times")


def test_get_entities_in_scene(scene_memory, sample_story):
    """Test getting entities in specific scene"""
    scene_memory.build_scene_graph(sample_story)
    
    # Get entities in first scene
    entities = scene_memory.get_entities_in_scene(0)
    
    assert len(entities) > 0
    # Should have at least 'seeker' and 'forest'
    entity_names = [e['name'] for e in entities]
    print(f"✅ Scene 0 entities: {entity_names}")


def test_entity_co_occurrences(scene_memory, sample_story):
    """Test finding entity co-occurrences"""
    scene_memory.build_scene_graph(sample_story)
    
    # Teacher and seeker should appear together in some scenes
    co_occur = scene_memory.find_entity_co_occurrences('teacher', 'seeker')
    
    assert len(co_occur) >= 1  # At least scene 2
    assert all('scene_index' in scene for scene in co_occur)
    
    print(f"✅ Teacher & Seeker co-occur in {len(co_occur)} scenes")


def test_entity_timeline(scene_memory, sample_story):
    """Test entity timeline extraction"""
    scene_memory.build_scene_graph(sample_story)
    
    timeline = scene_memory.get_entity_timeline('seeker')
    
    assert len(timeline) >= 2  # Seeker appears in multiple scenes
    assert timeline == sorted(timeline)  # Should be sorted
    assert all(isinstance(idx, int) for idx in timeline)
    
    print(f"✅ Seeker timeline: scenes {timeline}")


def test_graph_stats(scene_memory, sample_story):
    """Test graph statistics"""
    scene_memory.build_scene_graph(sample_story)
    
    stats = scene_memory.get_graph_stats()
    
    assert stats['total_scenes'] == 4
    assert stats['total_entities'] > 0
    assert stats['total_nodes'] > 0
    assert stats['total_edges'] > 0
    assert 'entity_types' in stats
    assert all(key in stats['entity_types'] for key in ['character', 'location', 'object'])
    
    print(f"✅ Graph stats: {stats}")


def test_scene_count(scene_memory, sample_story):
    """Test scene count"""
    scene_memory.build_scene_graph(sample_story)
    
    count = scene_memory.get_scene_count()
    assert count == 4
    
    print(f"✅ Scene count: {count}")


def test_entity_count(scene_memory, sample_story):
    """Test entity count"""
    scene_memory.build_scene_graph(sample_story)
    
    count = scene_memory.get_entity_count()
    assert count > 0
    
    print(f"✅ Entity count: {count}")


def test_entity_classification(scene_memory):
    """Test entity type classification"""
    # Test character classification
    assert scene_memory._classify_entity_type('seeker') == 'character'
    assert scene_memory._classify_entity_type('teacher') == 'character'
    
    # Test location classification
    assert scene_memory._classify_entity_type('forest') == 'location'
    assert scene_memory._classify_entity_type('mountain') == 'location'
    
    # Test object classification (default)
    assert scene_memory._classify_entity_type('wisdom') == 'object'
    
    print("✅ Entity classification working")


def test_singleton_pattern():
    """Test singleton pattern for scene memory"""
    instance1 = get_scene_memory()
    instance2 = get_scene_memory()
    
    assert instance1 is instance2  # Should be same instance
    
    print("✅ Singleton pattern working")


def test_export_to_json(scene_memory, sample_story, tmp_path):
    """Test JSON export functionality"""
    scene_memory.build_scene_graph(sample_story)
    
    output_file = tmp_path / "test_scene_graph.json"
    scene_memory.export_to_json(str(output_file))
    
    assert output_file.exists()
    
    # Verify JSON structure
    import json
    with open(output_file) as f:
        data = json.load(f)
    
    assert 'scenes' in data
    assert 'entities' in data
    assert 'transitions' in data
    assert 'statistics' in data
    assert len(data['scenes']) == 4
    
    print(f"✅ JSON export successful: {output_file}")


def test_cache_persistence(sample_story, tmp_path):
    """Test scene graph caching"""
    cache_dir = tmp_path / "cache"
    
    # Build graph and save to cache
    scene_memory1 = SceneMemoryCore(cache_dir=str(cache_dir))
    scene_memory1.build_scene_graph(sample_story)
    scene_count_1 = scene_memory1.get_scene_count()
    
    # Load from cache
    scene_memory2 = SceneMemoryCore(cache_dir=str(cache_dir))
    scene_count_2 = scene_memory2.get_scene_count()
    
    assert scene_count_2 == scene_count_1  # Should load from cache
    
    print("✅ Cache persistence working")


def test_empty_story():
    """Test handling empty story"""
    scene_memory = SceneMemoryCore()
    graph = scene_memory.build_scene_graph([])
    
    assert len(scene_memory.scenes) == 0
    assert len(scene_memory.entities) == 0
    
    print("✅ Empty story handled correctly")


def test_entity_not_found(scene_memory, sample_story):
    """Test querying non-existent entity"""
    scene_memory.build_scene_graph(sample_story)
    
    history = scene_memory.get_entity_history('nonexistent_entity')
    
    assert history['found'] == False
    assert 'message' in history
    
    print("✅ Non-existent entity handled correctly")


if __name__ == "__main__":
    """Run tests with detailed output"""
    print("=" * 60)
    print("SCENE MEMORY CORE - UNIT TESTS")
    print("=" * 60)
    
    # Run pytest
    pytest.main([__file__, "-v", "--tb=short"])
