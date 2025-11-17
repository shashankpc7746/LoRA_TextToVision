"""
Narrative Sequencer v1 - Day 3 of Task 11
Story beat parser, character arc tracking, and narrative continuity validation

Phase 2 Goal #4: Narrative Engine - COMPLETED HERE

Features:
- Story beat parser (setup, conflict, climax, resolution)
- Character arc tracking (introduction, development, transformation)
- Dialogue flow optimization
- Narrative continuity validation
- Pacing analysis and recommendations
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import pickle
import re


class StoryBeat(Enum):
    """Classic story structure beats"""
    SETUP = "setup"  # Introduction of characters, world, status quo
    INCITING_INCIDENT = "inciting_incident"  # Event that starts the story
    RISING_ACTION = "rising_action"  # Building tension and conflict
    CLIMAX = "climax"  # Peak of conflict and tension
    FALLING_ACTION = "falling_action"  # Consequences of climax
    RESOLUTION = "resolution"  # Conclusion, new equilibrium


class CharacterArcStage(Enum):
    """Stages in character development"""
    INTRODUCTION = "introduction"  # Character first appears
    ORDINARY_WORLD = "ordinary_world"  # Character in normal state
    CATALYST = "catalyst"  # Event that changes character
    DEVELOPMENT = "development"  # Character grows/changes
    TRANSFORMATION = "transformation"  # Character fundamentally changed
    NEW_EQUILIBRIUM = "new_equilibrium"  # Character in new state


class DialogueType(Enum):
    """Types of dialogue in narrative"""
    EXPOSITION = "exposition"  # Provides information
    CHARACTER_BUILDING = "character_building"  # Reveals character traits
    CONFLICT = "conflict"  # Argument or disagreement
    REVELATION = "revelation"  # Important information revealed
    EMOTIONAL = "emotional"  # Emotional exchange
    ACTION = "action"  # Accompanies action


@dataclass
class SceneBeat:
    """Represents a story beat in a specific scene"""
    scene_index: int
    beat_type: StoryBeat
    description: str
    tension_level: float  # 0.0-1.0
    pacing_speed: str  # 'slow', 'medium', 'fast'
    key_events: List[str] = field(default_factory=list)
    characters_involved: List[str] = field(default_factory=list)


@dataclass
class CharacterArc:
    """Tracks character development throughout story"""
    character_name: str
    arc_stage: CharacterArcStage
    scene_index: int
    emotional_state: str  # 'neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised'
    motivation: str
    relationships: Dict[str, str] = field(default_factory=dict)  # {other_char: relationship_type}
    growth_indicators: List[str] = field(default_factory=list)


@dataclass
class DialogueFlow:
    """Represents dialogue structure and flow"""
    scene_index: int
    dialogue_type: DialogueType
    speaker: str
    content: str
    subtext: Optional[str] = None  # Hidden meaning
    emotional_tone: str = "neutral"
    responses_to: Optional[int] = None  # Scene index of what this responds to


@dataclass
class NarrativeContinuity:
    """Tracks narrative consistency across scenes"""
    story_beats: List[SceneBeat]
    character_arcs: Dict[str, List[CharacterArc]]  # {character_name: [arcs]}
    dialogue_flows: List[DialogueFlow]
    continuity_issues: List[str] = field(default_factory=list)
    pacing_analysis: Dict[str, Any] = field(default_factory=dict)


class NarrativeSequencerV1:
    """
    Narrative intelligence for story understanding and sequencing
    
    Analyzes story structure, tracks character arcs, optimizes dialogue flow,
    and validates narrative continuity across scenes.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """Initialize narrative sequencer"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "narrative_sequencer.pkl"
        
        # Story structure tracking
        self.story_beats: List[SceneBeat] = []
        self.character_arcs: Dict[str, List[CharacterArc]] = {}
        self.dialogue_flows: List[DialogueFlow] = []
        
        # Analysis results
        self.continuity: Optional[NarrativeContinuity] = None
        
        # Story beat keywords for classification
        self.beat_keywords = {
            StoryBeat.SETUP: ['begins', 'starts', 'introduces', 'meets', 'lives', 'ordinary'],
            StoryBeat.INCITING_INCIDENT: ['suddenly', 'discovers', 'receives', 'encounters', 'happens'],
            StoryBeat.RISING_ACTION: ['challenges', 'struggles', 'faces', 'confronts', 'overcomes'],
            StoryBeat.CLIMAX: ['battle', 'confrontation', 'peak', 'crucial', 'decisive', 'final'],
            StoryBeat.FALLING_ACTION: ['aftermath', 'consequences', 'realizes', 'understands'],
            StoryBeat.RESOLUTION: ['finally', 'resolves', 'concludes', 'peace', 'harmony', 'ends']
        }
        
        # Dialogue indicators
        self.dialogue_indicators = {
            'says', 'asks', 'replies', 'answers', 'tells', 'speaks', 
            'whispers', 'shouts', 'declares', 'questions', 'responds'
        }
        
        # Emotional keywords
        self.emotion_keywords = {
            'happy': ['joy', 'happy', 'delighted', 'cheerful', 'pleased'],
            'sad': ['sad', 'sorrow', 'grief', 'melancholy', 'dejected'],
            'angry': ['angry', 'furious', 'enraged', 'irritated', 'frustrated'],
            'fearful': ['fear', 'afraid', 'scared', 'terrified', 'anxious'],
            'surprised': ['surprised', 'shocked', 'astonished', 'amazed', 'startled'],
            'neutral': ['calm', 'peaceful', 'serene', 'composed', 'steady']
        }
    
    def analyze_narrative(self, sentences: List[str], characters: Dict = None) -> NarrativeContinuity:
        """
        Analyze complete narrative structure
        
        Args:
            sentences: List of story sentences
            characters: Optional character data from story_context_parser
            
        Returns:
            NarrativeContinuity object with complete analysis
        """
        if not sentences:
            return NarrativeContinuity(
                story_beats=[],
                character_arcs={},
                dialogue_flows=[],
                continuity_issues=["Empty story provided"],
                pacing_analysis={}
            )
        
        # Parse story beats
        self.story_beats = self._parse_story_beats(sentences)
        
        # Track character arcs
        if characters:
            self.character_arcs = self._track_character_arcs(sentences, characters)
        
        # Analyze dialogue flow
        self.dialogue_flows = self._analyze_dialogue_flow(sentences, characters)
        
        # Validate continuity
        continuity_issues = self._validate_continuity()
        
        # Analyze pacing
        pacing_analysis = self._analyze_pacing()
        
        # Build continuity object
        self.continuity = NarrativeContinuity(
            story_beats=self.story_beats,
            character_arcs=self.character_arcs,
            dialogue_flows=self.dialogue_flows,
            continuity_issues=continuity_issues,
            pacing_analysis=pacing_analysis
        )
        
        # Cache the results
        self._save_cache()
        
        return self.continuity
    
    def _parse_story_beats(self, sentences: List[str]) -> List[SceneBeat]:
        """
        Parse story beats using classic story structure
        
        Uses keyword matching and position heuristics to identify beats
        """
        beats = []
        total_scenes = len(sentences)
        
        for idx, sentence in enumerate(sentences):
            # Calculate position ratio
            position = idx / max(total_scenes - 1, 1)
            
            # Default to rising action
            beat_type = StoryBeat.RISING_ACTION
            tension = 0.5
            pacing = 'medium'
            
            # Classify based on position and keywords
            sentence_lower = sentence.lower()
            
            # Setup (first ~20%)
            if position < 0.2:
                beat_type = StoryBeat.SETUP
                tension = 0.2
                pacing = 'slow'
                
                # Check for inciting incident
                if any(kw in sentence_lower for kw in self.beat_keywords[StoryBeat.INCITING_INCIDENT]):
                    beat_type = StoryBeat.INCITING_INCIDENT
                    tension = 0.4
                    pacing = 'medium'
            
            # Rising action (20%-70%)
            elif position < 0.7:
                beat_type = StoryBeat.RISING_ACTION
                # Increase tension as we approach climax
                tension = 0.3 + (position - 0.2) * 0.8  # 0.3 to 0.7
                pacing = 'medium'
                
                # Check for climax keywords even in this range
                if any(kw in sentence_lower for kw in self.beat_keywords[StoryBeat.CLIMAX]):
                    beat_type = StoryBeat.CLIMAX
                    tension = 1.0
                    pacing = 'fast'
            
            # Climax (70%-80%)
            elif position < 0.8:
                beat_type = StoryBeat.CLIMAX
                tension = 1.0
                pacing = 'fast'
            
            # Falling action (80%-90%)
            elif position < 0.9:
                beat_type = StoryBeat.FALLING_ACTION
                tension = 0.6
                pacing = 'medium'
            
            # Resolution (last ~10%)
            else:
                beat_type = StoryBeat.RESOLUTION
                tension = 0.3
                pacing = 'slow'
            
            # Extract key events (capitalized nouns/verbs)
            key_events = self._extract_key_events(sentence)
            
            # Extract characters mentioned
            characters_involved = self._extract_characters_from_sentence(sentence)
            
            beat = SceneBeat(
                scene_index=idx,
                beat_type=beat_type,
                description=sentence[:100],  # Truncate for storage
                tension_level=tension,
                pacing_speed=pacing,
                key_events=key_events,
                characters_involved=characters_involved
            )
            
            beats.append(beat)
        
        return beats
    
    def _track_character_arcs(self, sentences: List[str], characters: Dict) -> Dict[str, List[CharacterArc]]:
        """
        Track character development across scenes
        
        Args:
            sentences: Story sentences
            characters: Character data from story_context_parser
            
        Returns:
            Dictionary mapping character name to list of arc stages
        """
        arcs = {}
        
        for char_name, char_data in characters.items():
            char_arcs = []
            first_appearance = None
            
            for idx, sentence in enumerate(sentences):
                # Check if character appears in this scene
                if self._character_in_sentence(char_name, sentence):
                    if first_appearance is None:
                        first_appearance = idx
                        # Introduction stage
                        arc = CharacterArc(
                            character_name=char_name,
                            arc_stage=CharacterArcStage.INTRODUCTION,
                            scene_index=idx,
                            emotional_state=self._detect_emotion(sentence),
                            motivation=self._extract_motivation(sentence),
                            relationships={},
                            growth_indicators=[]
                        )
                        char_arcs.append(arc)
                    else:
                        # Determine arc stage based on position and context
                        arc_stage = self._determine_arc_stage(idx, len(sentences), sentence)
                        
                        arc = CharacterArc(
                            character_name=char_name,
                            arc_stage=arc_stage,
                            scene_index=idx,
                            emotional_state=self._detect_emotion(sentence),
                            motivation=self._extract_motivation(sentence),
                            relationships=self._extract_relationships(sentence, characters),
                            growth_indicators=self._detect_growth_indicators(sentence)
                        )
                        char_arcs.append(arc)
            
            if char_arcs:
                arcs[char_name] = char_arcs
        
        return arcs
    
    def _analyze_dialogue_flow(self, sentences: List[str], characters: Dict = None) -> List[DialogueFlow]:
        """
        Analyze dialogue structure and flow
        
        Detects dialogue, classifies type, tracks speaker, analyzes emotional tone
        """
        dialogues = []
        
        for idx, sentence in enumerate(sentences):
            # Check if sentence contains dialogue indicators
            if self._contains_dialogue(sentence):
                # Extract speaker
                speaker = self._extract_speaker(sentence, characters)
                
                # Classify dialogue type
                dialogue_type = self._classify_dialogue_type(sentence)
                
                # Detect emotional tone
                emotion = self._detect_emotion(sentence)
                
                # Extract subtext (implied meaning)
                subtext = self._extract_subtext(sentence)
                
                dialogue = DialogueFlow(
                    scene_index=idx,
                    dialogue_type=dialogue_type,
                    speaker=speaker,
                    content=sentence,
                    subtext=subtext,
                    emotional_tone=emotion,
                    responses_to=None  # Could be enhanced to track conversation threads
                )
                
                dialogues.append(dialogue)
        
        return dialogues
    
    def _validate_continuity(self) -> List[str]:
        """
        Validate narrative continuity
        
        Checks for:
        - Missing story beats
        - Character arc inconsistencies
        - Pacing issues
        - Logical gaps
        """
        issues = []
        
        # Check for essential story beats
        beat_types = {beat.beat_type for beat in self.story_beats}
        
        if StoryBeat.SETUP not in beat_types:
            issues.append("Missing SETUP beat - story may lack introduction")
        
        if StoryBeat.CLIMAX not in beat_types and len(self.story_beats) > 3:
            issues.append("Missing CLIMAX beat - story may lack peak tension")
        
        if StoryBeat.RESOLUTION not in beat_types and len(self.story_beats) > 2:
            issues.append("Missing RESOLUTION beat - story may lack conclusion")
        
        # Check character arc completeness
        for char_name, arcs in self.character_arcs.items():
            if not arcs:
                issues.append(f"Character '{char_name}' has no tracked arc")
                continue
            
            arc_stages = {arc.arc_stage for arc in arcs}
            
            if CharacterArcStage.INTRODUCTION not in arc_stages:
                issues.append(f"Character '{char_name}' missing INTRODUCTION")
        
        # Check pacing distribution
        pacing_counts = {'slow': 0, 'medium': 0, 'fast': 0}
        for beat in self.story_beats:
            pacing_counts[beat.pacing_speed] += 1
        
        total_beats = len(self.story_beats)
        if total_beats > 0:
            # Check if pacing is too monotonous
            for pace, count in pacing_counts.items():
                if count / total_beats > 0.8:
                    issues.append(f"Pacing too monotonous - {pace} dominates {count}/{total_beats} scenes")
        
        return issues
    
    def _analyze_pacing(self) -> Dict[str, Any]:
        """
        Analyze story pacing
        
        Returns pacing statistics and recommendations
        """
        if not self.story_beats:
            return {}
        
        pacing_distribution = {'slow': 0, 'medium': 0, 'fast': 0}
        tension_curve = []
        
        for beat in self.story_beats:
            pacing_distribution[beat.pacing_speed] += 1
            tension_curve.append(beat.tension_level)
        
        total_beats = len(self.story_beats)
        
        analysis = {
            'total_scenes': total_beats,
            'pacing_distribution': {
                pace: {'count': count, 'percentage': (count / total_beats) * 100}
                for pace, count in pacing_distribution.items()
            },
            'tension_curve': tension_curve,
            'average_tension': sum(tension_curve) / len(tension_curve) if tension_curve else 0,
            'peak_tension': max(tension_curve) if tension_curve else 0,
            'recommendations': []
        }
        
        # Generate recommendations
        slow_pct = pacing_distribution['slow'] / total_beats
        fast_pct = pacing_distribution['fast'] / total_beats
        
        if slow_pct > 0.6:
            analysis['recommendations'].append("Consider adding more fast-paced scenes for variety")
        
        if fast_pct > 0.6:
            analysis['recommendations'].append("Consider adding slower moments for breathing room")
        
        if analysis['average_tension'] < 0.3:
            analysis['recommendations'].append("Overall tension is low - consider raising stakes")
        
        return analysis
    
    # Helper methods
    
    def _extract_key_events(self, sentence: str) -> List[str]:
        """Extract key events from sentence (important actions/nouns)"""
        # Simple extraction - could be enhanced with NLP
        words = sentence.split()
        key_events = []
        
        for word in words:
            # Look for capitalized words (likely important) or action verbs
            if word[0].isupper() and len(word) > 3:
                key_events.append(word.strip('.,!?;:'))
        
        return key_events[:5]  # Limit to 5 key events
    
    def _extract_characters_from_sentence(self, sentence: str) -> List[str]:
        """Extract character names from sentence"""
        # Simple extraction - in production would use character data
        characters = []
        common_characters = ['seeker', 'teacher', 'master', 'student', 'guide', 'sage']
        
        sentence_lower = sentence.lower()
        for char in common_characters:
            if char in sentence_lower:
                characters.append(char)
        
        return characters
    
    def _character_in_sentence(self, char_name: str, sentence: str) -> bool:
        """Check if character is mentioned in sentence"""
        sentence_lower = sentence.lower()
        char_lower = char_name.lower()
        
        # Check direct mention
        if char_lower in sentence_lower:
            return True
        
        # Check pronouns (if this is continuation)
        pronouns = ['he', 'she', 'they', 'him', 'her', 'them']
        if any(pronoun in sentence_lower for pronoun in pronouns):
            return True  # Could be more sophisticated
        
        return False
    
    def _detect_emotion(self, sentence: str) -> str:
        """Detect emotional tone of sentence"""
        sentence_lower = sentence.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            if any(kw in sentence_lower for kw in keywords):
                return emotion
        
        return 'neutral'
    
    def _extract_motivation(self, sentence: str) -> str:
        """Extract character motivation from sentence"""
        # Look for goal-oriented keywords
        motivation_keywords = {
            'seeks', 'wants', 'desires', 'needs', 'hopes', 'wishes',
            'intends', 'plans', 'aims', 'strives', 'pursues'
        }
        
        sentence_lower = sentence.lower()
        for keyword in motivation_keywords:
            if keyword in sentence_lower:
                # Extract context around keyword
                idx = sentence_lower.index(keyword)
                context = sentence[max(0, idx-20):min(len(sentence), idx+50)]
                return context.strip()
        
        return "unknown"
    
    def _determine_arc_stage(self, scene_idx: int, total_scenes: int, sentence: str) -> CharacterArcStage:
        """Determine character arc stage based on position and content"""
        position = scene_idx / max(total_scenes - 1, 1)
        sentence_lower = sentence.lower()
        
        # Transformation keywords
        if any(kw in sentence_lower for kw in ['transformed', 'changed', 'became', 'realized']):
            return CharacterArcStage.TRANSFORMATION
        
        # Catalyst keywords
        if any(kw in sentence_lower for kw in ['suddenly', 'discovered', 'encountered', 'met']):
            return CharacterArcStage.CATALYST
        
        # Position-based stages
        if position < 0.3:
            return CharacterArcStage.ORDINARY_WORLD
        elif position < 0.7:
            return CharacterArcStage.DEVELOPMENT
        elif position < 0.9:
            return CharacterArcStage.TRANSFORMATION
        else:
            return CharacterArcStage.NEW_EQUILIBRIUM
    
    def _extract_relationships(self, sentence: str, characters: Dict) -> Dict[str, str]:
        """Extract character relationships from sentence"""
        relationships = {}
        
        # Relationship keywords
        rel_keywords = {
            'teacher': 'mentor',
            'student': 'learner',
            'friend': 'companion',
            'enemy': 'adversary',
            'master': 'mentor',
            'guide': 'mentor'
        }
        
        sentence_lower = sentence.lower()
        for keyword, rel_type in rel_keywords.items():
            if keyword in sentence_lower:
                relationships[keyword] = rel_type
        
        return relationships
    
    def _detect_growth_indicators(self, sentence: str) -> List[str]:
        """Detect indicators of character growth"""
        indicators = []
        
        growth_keywords = [
            'learned', 'understood', 'realized', 'discovered',
            'grew', 'changed', 'transformed', 'evolved',
            'mastered', 'overcame', 'achieved'
        ]
        
        sentence_lower = sentence.lower()
        for keyword in growth_keywords:
            if keyword in sentence_lower:
                indicators.append(keyword)
        
        return indicators
    
    def _contains_dialogue(self, sentence: str) -> bool:
        """Check if sentence contains dialogue"""
        sentence_lower = sentence.lower()
        
        # Check for dialogue indicators
        if any(indicator in sentence_lower for indicator in self.dialogue_indicators):
            return True
        
        # Check for quotation marks
        if '"' in sentence or "'" in sentence:
            return True
        
        return False
    
    def _extract_speaker(self, sentence: str, characters: Dict = None) -> str:
        """Extract speaker from dialogue sentence"""
        sentence_lower = sentence.lower()
        
        # Look for common patterns: "X says/asks/tells"
        for indicator in self.dialogue_indicators:
            if indicator in sentence_lower:
                # Get word before indicator
                words = sentence_lower.split()
                try:
                    idx = words.index(indicator)
                    if idx > 0:
                        return words[idx - 1].strip('.,!?;:')
                except ValueError:
                    pass
        
        # If characters provided, check for mentions
        if characters:
            for char_name in characters.keys():
                if char_name.lower() in sentence_lower:
                    return char_name
        
        return "unknown"
    
    def _classify_dialogue_type(self, sentence: str) -> DialogueType:
        """Classify type of dialogue"""
        sentence_lower = sentence.lower()
        
        # Exposition
        if any(kw in sentence_lower for kw in ['explains', 'tells', 'describes', 'informs']):
            return DialogueType.EXPOSITION
        
        # Conflict
        if any(kw in sentence_lower for kw in ['argues', 'disagrees', 'confronts', 'challenges']):
            return DialogueType.CONFLICT
        
        # Revelation
        if any(kw in sentence_lower for kw in ['reveals', 'discovers', 'realizes', 'truth']):
            return DialogueType.REVELATION
        
        # Emotional
        if self._detect_emotion(sentence) != 'neutral':
            return DialogueType.EMOTIONAL
        
        # Character building (default for dialogue)
        return DialogueType.CHARACTER_BUILDING
    
    def _extract_subtext(self, sentence: str) -> Optional[str]:
        """Extract implied meaning or subtext from dialogue"""
        # This is a placeholder - real implementation would use more sophisticated NLP
        sentence_lower = sentence.lower()
        
        # Look for sarcasm/irony indicators
        if any(kw in sentence_lower for kw in ['really', 'sure', 'obviously', 'clearly']):
            return "possible sarcasm or emphasis"
        
        return None
    
    def _save_cache(self):
        """Save narrative analysis to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.continuity, f)
        except Exception as e:
            print(f"Warning: Could not save narrative cache: {e}")
    
    def _load_cache(self) -> bool:
        """Load narrative analysis from cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.continuity = pickle.load(f)
                return True
        except Exception as e:
            print(f"Warning: Could not load narrative cache: {e}")
        return False
    
    def export_to_json(self, output_path: str):
        """Export narrative analysis to JSON for debugging/visualization"""
        if not self.continuity:
            return
        
        data = {
            'story_beats': [
                {
                    'scene_index': beat.scene_index,
                    'beat_type': beat.beat_type.value,
                    'description': beat.description,
                    'tension_level': beat.tension_level,
                    'pacing_speed': beat.pacing_speed,
                    'key_events': beat.key_events,
                    'characters_involved': beat.characters_involved
                }
                for beat in self.continuity.story_beats
            ],
            'character_arcs': {
                char_name: [
                    {
                        'scene_index': arc.scene_index,
                        'arc_stage': arc.arc_stage.value,
                        'emotional_state': arc.emotional_state,
                        'motivation': arc.motivation,
                        'relationships': arc.relationships,
                        'growth_indicators': arc.growth_indicators
                    }
                    for arc in arcs
                ]
                for char_name, arcs in self.continuity.character_arcs.items()
            },
            'dialogue_flows': [
                {
                    'scene_index': dlg.scene_index,
                    'dialogue_type': dlg.dialogue_type.value,
                    'speaker': dlg.speaker,
                    'emotional_tone': dlg.emotional_tone,
                    'subtext': dlg.subtext
                }
                for dlg in self.continuity.dialogue_flows
            ],
            'continuity_issues': self.continuity.continuity_issues,
            'pacing_analysis': self.continuity.pacing_analysis
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Singleton instance
_narrative_sequencer_instance = None

def get_narrative_sequencer() -> NarrativeSequencerV1:
    """Get global narrative sequencer instance (singleton pattern)"""
    global _narrative_sequencer_instance
    if _narrative_sequencer_instance is None:
        _narrative_sequencer_instance = NarrativeSequencerV1()
    return _narrative_sequencer_instance


if __name__ == "__main__":
    # Example usage
    print("=== Narrative Sequencer v1 Demo ===\n")
    
    # Sample story
    story = [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests where ancient sages meditated",
        "The seeker encounters a wise teacher at an old temple",
        "Together they discuss the nature of reality and consciousness",
        "The seeker realizes the truth has been within her all along"
    ]
    
    # Sample characters
    characters = {
        'seeker': {'gender': 'female', 'role': 'protagonist'},
        'teacher': {'gender': 'male', 'role': 'mentor'}
    }
    
    # Analyze narrative
    sequencer = get_narrative_sequencer()
    continuity = sequencer.analyze_narrative(story, characters)
    
    # Display results
    print(f"📖 Story Beats Identified: {len(continuity.story_beats)}")
    for beat in continuity.story_beats:
        print(f"  Scene {beat.scene_index}: {beat.beat_type.value} (tension: {beat.tension_level:.2f})")
    
    print(f"\n👤 Character Arcs: {len(continuity.character_arcs)}")
    for char_name, arcs in continuity.character_arcs.items():
        print(f"  {char_name}: {len(arcs)} stages tracked")
        for arc in arcs:
            print(f"    Scene {arc.scene_index}: {arc.arc_stage.value} ({arc.emotional_state})")
    
    print(f"\n💬 Dialogue Flows: {len(continuity.dialogue_flows)}")
    for dlg in continuity.dialogue_flows:
        print(f"  Scene {dlg.scene_index}: {dlg.speaker} - {dlg.dialogue_type.value}")
    
    print(f"\n⚠️ Continuity Issues: {len(continuity.continuity_issues)}")
    for issue in continuity.continuity_issues:
        print(f"  - {issue}")
    
    print(f"\n📊 Pacing Analysis:")
    pacing = continuity.pacing_analysis
    print(f"  Average tension: {pacing.get('average_tension', 0):.2f}")
    print(f"  Peak tension: {pacing.get('peak_tension', 0):.2f}")
    for rec in pacing.get('recommendations', []):
        print(f"  💡 {rec}")
    
    # Export to JSON
    sequencer.export_to_json("narrative_analysis_example.json")
    print(f"\n✅ Exported analysis to narrative_analysis_example.json")
