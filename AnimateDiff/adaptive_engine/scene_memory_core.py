"""
Scene Memory Core - Day 2 of Task 11
Temporal scene graph with entity tracking and cross-scene relationships

Phase 2 Goal #1: Scene Graph Module - COMPLETED HERE

Features:
- Temporal scene graph with NetworkX
- Entity persistence across scenes
- Cross-scene relationship tracking
- Scene transition metadata
- Graph query API for entity history and transitions
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import pickle


@dataclass
class SceneNode:
    """Represents a single scene in the story"""
    scene_id: str
    scene_index: int
    text: str
    timestamp: float
    duration: float = 0.0
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode:
    """Represents an entity (character, object, location) in the story"""
    entity_id: str
    entity_type: str  # 'character', 'object', 'location'
    name: str
    first_appearance: int  # scene index
    last_appearance: int  # scene index
    total_appearances: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneTransition:
    """Represents a transition between two scenes"""
    from_scene: str
    to_scene: str
    transition_type: str = "temporal_next"  # 'temporal_next', 'location_change', 'time_jump'
    duration: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneMemoryCore:
    """
    Scene Memory Core - Temporal scene graph for story understanding
    
    Phase 2 Goal #1: Standalone scene graph module with:
    - Entity persistence across scenes
    - Temporal relationship tracking
    - Queryable graph structure
    - Cross-scene entity linking
    
    Integration Bonus (Task 11):
    - Works with identity_memory (Day 1) for character tracking
    - Works with story_context_parser (Day 1) for entity extraction
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize scene memory core"""
        self.graph = nx.DiGraph()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "scene_memory.pkl"
        
        # Entity tracking
        self.entities: Dict[str, EntityNode] = {}
        self.scenes: Dict[str, SceneNode] = {}
        
        # Load from cache if exists
        self._load_cache()
    
    def build_scene_graph(self, 
                         story_sentences: List[str],
                         characters: Optional[Dict] = None) -> nx.DiGraph:
        """
        Build complete scene graph from story
        
        Args:
            story_sentences: List of sentences in the story
            characters: Optional character dict from story_context_parser
        
        Returns:
            NetworkX directed graph with scenes and entities
        """
        print(f"🎬 Building scene graph for {len(story_sentences)} scenes...")
        
        # Clear existing graph
        self.graph.clear()
        self.scenes.clear()
        self.entities.clear()
        
        # Process each sentence as a scene
        for idx, sentence in enumerate(story_sentences):
            scene_id = f"scene_{idx}"
            
            # Create scene node
            scene = SceneNode(
                scene_id=scene_id,
                scene_index=idx,
                text=sentence,
                timestamp=idx * 3.0,  # Assuming 3 seconds per scene
                duration=3.0
            )
            
            # Extract entities from sentence
            entities = self._extract_entities_from_sentence(sentence, characters)
            scene.entities = entities
            
            # Add scene node to graph
            self.graph.add_node(
                scene_id,
                node_type='scene',
                scene_index=idx,
                text=sentence,
                timestamp=scene.timestamp,
                duration=scene.duration,
                entities=entities
            )
            
            self.scenes[scene_id] = scene
            
            # Track entity appearances
            for entity_name in entities:
                self._update_entity_tracking(entity_name, idx, scene_id)
            
            # Link temporal progression (scene A → scene B)
            if idx > 0:
                prev_scene_id = f"scene_{idx-1}"
                self._add_scene_transition(
                    prev_scene_id,
                    scene_id,
                    transition_type="temporal_next"
                )
            
            print(f"   ✅ Scene {idx}: {len(entities)} entities - {sentence[:50]}...")
        
        # Add entity nodes to graph
        for entity_id, entity in self.entities.items():
            self.graph.add_node(
                entity_id,
                node_type='entity',
                entity_type=entity.entity_type,
                name=entity.name,
                first_appearance=entity.first_appearance,
                last_appearance=entity.last_appearance,
                total_appearances=entity.total_appearances
            )
            
            # Link entities to scenes they appear in
            for scene_id, scene in self.scenes.items():
                if entity.name in scene.entities:
                    self.graph.add_edge(
                        entity_id,
                        scene_id,
                        edge_type='appears_in'
                    )
        
        print(f"✅ Scene graph built: {len(self.scenes)} scenes, {len(self.entities)} entities")
        
        # Save to cache
        self._save_cache()
        
        return self.graph
    
    def _extract_entities_from_sentence(self, 
                                       sentence: str,
                                       characters: Optional[Dict] = None) -> List[str]:
        """Extract entity names from sentence"""
        entities = []
        
        # Use provided characters if available
        if characters:
            for char_name in characters.keys():
                if char_name.lower() in sentence.lower():
                    entities.append(char_name)
        
        # Simple keyword extraction for common entities
        entity_keywords = {
            'character': ['seeker', 'teacher', 'student', 'monk', 'sage', 'master', 'disciple'],
            'location': ['forest', 'mountain', 'temple', 'river', 'valley', 'cave', 'path'],
            'object': ['wisdom', 'knowledge', 'book', 'scroll', 'lantern', 'staff']
        }
        
        sentence_lower = sentence.lower()
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower and keyword not in entities:
                    entities.append(keyword)
        
        return entities
    
    def _update_entity_tracking(self, entity_name: str, scene_index: int, scene_id: str):
        """Update entity tracking information"""
        entity_id = f"entity_{entity_name}"
        
        if entity_id not in self.entities:
            # New entity
            entity_type = self._classify_entity_type(entity_name)
            self.entities[entity_id] = EntityNode(
                entity_id=entity_id,
                entity_type=entity_type,
                name=entity_name,
                first_appearance=scene_index,
                last_appearance=scene_index,
                total_appearances=1
            )
        else:
            # Existing entity - update appearances
            entity = self.entities[entity_id]
            entity.last_appearance = scene_index
            entity.total_appearances += 1
    
    def _classify_entity_type(self, entity_name: str) -> str:
        """Classify entity as character, object, or location"""
        entity_lower = entity_name.lower()
        
        # Character indicators
        character_words = ['seeker', 'teacher', 'student', 'monk', 'sage', 'master', 'disciple', 'he', 'she']
        if any(word in entity_lower for word in character_words):
            return 'character'
        
        # Location indicators
        location_words = ['forest', 'mountain', 'temple', 'river', 'valley', 'cave', 'path', 'garden']
        if any(word in entity_lower for word in location_words):
            return 'location'
        
        # Default to object
        return 'object'
    
    def _add_scene_transition(self, 
                             from_scene: str,
                             to_scene: str,
                             transition_type: str = "temporal_next"):
        """Add transition edge between scenes"""
        self.graph.add_edge(
            from_scene,
            to_scene,
            edge_type='transition',
            transition_type=transition_type
        )
    
    # ========== QUERY API ==========
    
    def get_entity_history(self, entity_name: str) -> Dict[str, Any]:
        """
        Get complete appearance history for an entity
        
        Returns:
            Dict with first/last appearance, scenes appeared in, total count
        """
        entity_id = f"entity_{entity_name}"
        
        if entity_id not in self.entities:
            return {
                'found': False,
                'entity_name': entity_name,
                'message': 'Entity not found in scene graph'
            }
        
        entity = self.entities[entity_id]
        
        # Find all scenes this entity appears in
        scenes_appeared = []
        for scene_id, scene in self.scenes.items():
            if entity_name in scene.entities:
                scenes_appeared.append({
                    'scene_id': scene_id,
                    'scene_index': scene.scene_index,
                    'text': scene.text,
                    'timestamp': scene.timestamp
                })
        
        return {
            'found': True,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'entity_type': entity.entity_type,
            'first_appearance': entity.first_appearance,
            'last_appearance': entity.last_appearance,
            'total_appearances': entity.total_appearances,
            'scenes': scenes_appeared
        }
    
    def get_scene_transitions(self, from_index: Optional[int] = None) -> List[Dict]:
        """
        Get scene transition information
        
        Args:
            from_index: If specified, only get transitions from this scene
        
        Returns:
            List of transition dicts
        """
        transitions = []
        
        for edge in self.graph.edges(data=True):
            from_node, to_node, data = edge
            
            # Only process transition edges
            if data.get('edge_type') != 'transition':
                continue
            
            # Filter by from_index if specified
            if from_index is not None:
                from_scene_idx = self.graph.nodes[from_node].get('scene_index')
                if from_scene_idx != from_index:
                    continue
            
            transitions.append({
                'from_scene': from_node,
                'to_scene': to_node,
                'transition_type': data.get('transition_type', 'temporal_next'),
                'from_text': self.graph.nodes[from_node].get('text', ''),
                'to_text': self.graph.nodes[to_node].get('text', '')
            })
        
        return transitions
    
    def get_entities_in_scene(self, scene_index: int) -> List[Dict]:
        """Get all entities that appear in a specific scene"""
        scene_id = f"scene_{scene_index}"
        
        if scene_id not in self.scenes:
            return []
        
        scene = self.scenes[scene_id]
        
        entities_info = []
        for entity_name in scene.entities:
            entity_id = f"entity_{entity_name}"
            if entity_id in self.entities:
                entity = self.entities[entity_id]
                entities_info.append({
                    'name': entity.name,
                    'type': entity.entity_type,
                    'is_first_appearance': entity.first_appearance == scene_index,
                    'is_last_appearance': entity.last_appearance == scene_index,
                    'total_appearances': entity.total_appearances
                })
        
        return entities_info
    
    def get_scene_count(self) -> int:
        """Get total number of scenes in graph"""
        return len(self.scenes)
    
    def get_entity_count(self) -> int:
        """Get total number of unique entities"""
        return len(self.entities)
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        return {
            'total_scenes': len(self.scenes),
            'total_entities': len(self.entities),
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'entity_types': {
                'character': sum(1 for e in self.entities.values() if e.entity_type == 'character'),
                'location': sum(1 for e in self.entities.values() if e.entity_type == 'location'),
                'object': sum(1 for e in self.entities.values() if e.entity_type == 'object')
            },
            'avg_entities_per_scene': len(self.entities) / len(self.scenes) if self.scenes else 0
        }
    
    def find_entity_co_occurrences(self, entity1: str, entity2: str) -> List[Dict]:
        """Find scenes where two entities appear together"""
        co_occurrences = []
        
        for scene_id, scene in self.scenes.items():
            if entity1 in scene.entities and entity2 in scene.entities:
                co_occurrences.append({
                    'scene_id': scene_id,
                    'scene_index': scene.scene_index,
                    'text': scene.text,
                    'timestamp': scene.timestamp
                })
        
        return co_occurrences
    
    def get_entity_timeline(self, entity_name: str) -> List[int]:
        """Get list of scene indices where entity appears (timeline)"""
        timeline = []
        
        for scene_id, scene in self.scenes.items():
            if entity_name in scene.entities:
                timeline.append(scene.scene_index)
        
        return sorted(timeline)
    
    # ========== PERSISTENCE ==========
    
    def _save_cache(self):
        """Save scene graph to cache"""
        try:
            cache_data = {
                'graph': self.graph,
                'entities': self.entities,
                'scenes': self.scenes,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"   💾 Scene graph cached to {self.cache_file}")
        except Exception as e:
            print(f"   ⚠️ Cache save failed: {e}")
    
    def _load_cache(self):
        """Load scene graph from cache"""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.graph = cache_data.get('graph', nx.DiGraph())
            self.entities = cache_data.get('entities', {})
            self.scenes = cache_data.get('scenes', {})
            
            print(f"   📂 Loaded scene graph from cache ({len(self.scenes)} scenes)")
        except Exception as e:
            print(f"   ⚠️ Cache load failed: {e}")
            self.graph = nx.DiGraph()
            self.entities = {}
            self.scenes = {}
    
    def export_to_json(self, output_path: str):
        """Export scene graph to JSON format"""
        export_data = {
            'scenes': [
                {
                    'scene_id': scene.scene_id,
                    'scene_index': scene.scene_index,
                    'text': scene.text,
                    'timestamp': scene.timestamp,
                    'duration': scene.duration,
                    'entities': scene.entities
                }
                for scene in self.scenes.values()
            ],
            'entities': [
                {
                    'entity_id': entity.entity_id,
                    'name': entity.name,
                    'type': entity.entity_type,
                    'first_appearance': entity.first_appearance,
                    'last_appearance': entity.last_appearance,
                    'total_appearances': entity.total_appearances
                }
                for entity in self.entities.values()
            ],
            'transitions': self.get_scene_transitions(),
            'statistics': self.get_graph_stats()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Scene graph exported to {output_path}")


# ========== SINGLETON INSTANCE ==========

_scene_memory_instance = None

def get_scene_memory() -> SceneMemoryCore:
    """Get global scene memory instance (singleton pattern)"""
    global _scene_memory_instance
    if _scene_memory_instance is None:
        _scene_memory_instance = SceneMemoryCore()
    return _scene_memory_instance


# ========== QUICK USAGE EXAMPLE ==========

if __name__ == "__main__":
    # Example usage
    story = [
        "A young seeker begins her journey through the misty forest.",
        "She encounters a wise teacher at the mountain temple.",
        "The teacher shares ancient wisdom with the seeker.",
        "Together they walk the sacred path through the valley."
    ]
    
    # Build scene graph
    scene_memory = get_scene_memory()
    graph = scene_memory.build_scene_graph(story)
    
    # Query examples
    print("\n" + "="*60)
    print("SCENE GRAPH QUERY EXAMPLES")
    print("="*60)
    
    # Entity history
    seeker_history = scene_memory.get_entity_history('seeker')
    print(f"\n📊 Seeker appears in {seeker_history['total_appearances']} scenes")
    
    # Scene transitions
    transitions = scene_memory.get_scene_transitions()
    print(f"\n🎬 {len(transitions)} scene transitions")
    
    # Co-occurrences
    co_occur = scene_memory.find_entity_co_occurrences('seeker', 'teacher')
    print(f"\n👥 Seeker & Teacher appear together in {len(co_occur)} scenes")
    
    # Graph stats
    stats = scene_memory.get_graph_stats()
    print(f"\n📈 Graph: {stats['total_scenes']} scenes, {stats['total_entities']} entities")
    
    # Export to JSON
    scene_memory.export_to_json("scene_graph_example.json")
    print("\n✅ Scene Memory Core demonstration complete!")
