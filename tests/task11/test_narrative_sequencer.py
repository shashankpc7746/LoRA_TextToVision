"""
Test suite for Narrative Sequencer v1
Comprehensive tests for story beat parsing, character arcs, and narrative continuity
"""

import pytest
import sys
from pathlib import Path

# Add AnimateDiff to path
animatediff_path = Path(__file__).parent.parent.parent / "AnimateDiff"
sys.path.insert(0, str(animatediff_path))

from adaptive_engine.narrative_sequencer_v1 import (
    NarrativeSequencerV1,
    get_narrative_sequencer,
    StoryBeat,
    CharacterArcStage,
    DialogueType,
    SceneBeat,
    CharacterArc,
    DialogueFlow,
    NarrativeContinuity
)


@pytest.fixture
def sample_story():
    """Sample story for testing"""
    return [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests where ancient sages meditated",
        "The seeker encounters a wise teacher at an old temple",
        "Together they discuss the nature of reality and consciousness",
        "The seeker realizes the truth has been within her all along"
    ]


@pytest.fixture
def sample_characters():
    """Sample character data"""
    return {
        'seeker': {'gender': 'female', 'role': 'protagonist'},
        'teacher': {'gender': 'male', 'role': 'mentor'}
    }


@pytest.fixture
def narrative_sequencer():
    """Fresh narrative sequencer instance"""
    return NarrativeSequencerV1()


def test_narrative_sequencer_initialization(narrative_sequencer):
    """Test narrative sequencer initializes correctly"""
    assert narrative_sequencer is not None
    assert narrative_sequencer.story_beats == []
    assert narrative_sequencer.character_arcs == {}
    assert narrative_sequencer.dialogue_flows == []
    assert narrative_sequencer.continuity is None
    print("✅ Narrative sequencer initializes correctly")


def test_analyze_narrative(narrative_sequencer, sample_story, sample_characters):
    """Test complete narrative analysis"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, sample_characters)
    
    assert isinstance(continuity, NarrativeContinuity)
    assert len(continuity.story_beats) == 5
    assert len(continuity.character_arcs) >= 1  # At least 'seeker'
    assert isinstance(continuity.pacing_analysis, dict)
    assert len(continuity.pacing_analysis) > 0
    print("✅ Narrative analysis completes successfully")


def test_story_beat_parsing(narrative_sequencer, sample_story):
    """Test story beat classification"""
    beats = narrative_sequencer._parse_story_beats(sample_story)
    
    assert len(beats) == 5
    
    # Check beat types are assigned
    beat_types = [beat.beat_type for beat in beats]
    assert StoryBeat.SETUP in beat_types  # First scene should be setup
    assert StoryBeat.RESOLUTION in beat_types  # Last scene should be resolution
    
    # Check tension levels
    for beat in beats:
        assert 0.0 <= beat.tension_level <= 1.0
    
    # Check pacing speeds
    pacing_speeds = {beat.pacing_speed for beat in beats}
    assert pacing_speeds.issubset({'slow', 'medium', 'fast'})
    
    print("✅ Story beats parsed correctly")


def test_story_beat_progression(narrative_sequencer, sample_story):
    """Test that story beats follow logical progression"""
    beats = narrative_sequencer._parse_story_beats(sample_story)
    
    # First beat should be setup
    assert beats[0].beat_type == StoryBeat.SETUP
    assert beats[0].tension_level < 0.5
    
    # Last beat should be resolution
    assert beats[-1].beat_type == StoryBeat.RESOLUTION
    assert beats[-1].pacing_speed in ['slow', 'medium']
    
    # Tension should generally increase then decrease
    tension_curve = [beat.tension_level for beat in beats]
    max_tension_idx = tension_curve.index(max(tension_curve))
    
    # Peak tension should not be at the very beginning
    assert max_tension_idx > 0
    
    print("✅ Story beat progression is logical")


def test_character_arc_tracking(narrative_sequencer, sample_story, sample_characters):
    """Test character arc tracking across scenes"""
    arcs = narrative_sequencer._track_character_arcs(sample_story, sample_characters)
    
    assert 'seeker' in arcs
    seeker_arcs = arcs['seeker']
    
    # Seeker appears in multiple scenes
    assert len(seeker_arcs) >= 2
    
    # First appearance should be introduction
    assert seeker_arcs[0].arc_stage == CharacterArcStage.INTRODUCTION
    assert seeker_arcs[0].scene_index == 0
    
    # Each arc should have required fields
    for arc in seeker_arcs:
        assert isinstance(arc.emotional_state, str)
        assert isinstance(arc.motivation, str)
        assert isinstance(arc.scene_index, int)
    
    print("✅ Character arcs tracked correctly")


def test_character_arc_stages(narrative_sequencer, sample_story, sample_characters):
    """Test character arc stage classification"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, sample_characters)
    
    seeker_arcs = continuity.character_arcs.get('seeker', [])
    assert len(seeker_arcs) > 0
    
    # Check arc stages progress logically
    arc_stages = [arc.arc_stage for arc in seeker_arcs]
    
    # Introduction should come first
    assert arc_stages[0] == CharacterArcStage.INTRODUCTION
    
    # Should have some development/transformation stages
    stage_types = set(arc_stages)
    assert len(stage_types) >= 2  # At least 2 different stages
    
    print("✅ Character arc stages classified correctly")


def test_dialogue_flow_detection(narrative_sequencer):
    """Test dialogue detection and analysis"""
    dialogue_sentences = [
        "The teacher says to the student, 'Listen carefully'",
        "She asks, 'What is the meaning of life?'",
        "The master replies with wisdom"
    ]
    
    dialogues = narrative_sequencer._analyze_dialogue_flow(dialogue_sentences)
    
    # Should detect dialogue indicators
    assert len(dialogues) >= 1
    
    for dlg in dialogues:
        assert isinstance(dlg.dialogue_type, DialogueType)
        assert isinstance(dlg.speaker, str)
        assert isinstance(dlg.emotional_tone, str)
    
    print("✅ Dialogue flow detected and analyzed")


def test_emotion_detection(narrative_sequencer):
    """Test emotional tone detection"""
    happy_sentence = "The seeker felt joyful and delighted"
    sad_sentence = "She experienced deep sorrow and grief"
    angry_sentence = "The warrior became furious and enraged"
    neutral_sentence = "They walked through the forest"
    
    assert narrative_sequencer._detect_emotion(happy_sentence) == 'happy'
    assert narrative_sequencer._detect_emotion(sad_sentence) == 'sad'
    assert narrative_sequencer._detect_emotion(angry_sentence) == 'angry'
    assert narrative_sequencer._detect_emotion(neutral_sentence) == 'neutral'
    
    print("✅ Emotion detection working correctly")


def test_continuity_validation(narrative_sequencer, sample_story, sample_characters):
    """Test narrative continuity validation"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, sample_characters)
    
    # Should identify continuity issues (or lack thereof)
    assert isinstance(continuity.continuity_issues, list)
    
    # Well-formed story should have few/no issues
    assert len(continuity.continuity_issues) <= 2
    
    print("✅ Continuity validation working")


def test_continuity_validation_incomplete_story(narrative_sequencer):
    """Test continuity validation catches incomplete stories"""
    incomplete_story = ["The hero arrives"]  # No climax, no resolution
    
    continuity = narrative_sequencer.analyze_narrative(incomplete_story, {})
    
    # Should flag missing beats or have issues (story is very short)
    issues = continuity.continuity_issues
    # With only 1 scene, system should detect some issue
    assert len(issues) >= 1  # At least one issue should be flagged
    
    print("✅ Continuity validation catches incomplete stories")


def test_pacing_analysis(narrative_sequencer, sample_story, sample_characters):
    """Test pacing analysis and recommendations"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, sample_characters)
    
    pacing = continuity.pacing_analysis
    
    # Should have key metrics
    assert 'total_scenes' in pacing
    assert 'pacing_distribution' in pacing
    assert 'tension_curve' in pacing
    assert 'average_tension' in pacing
    assert 'recommendations' in pacing
    
    # Metrics should be valid
    assert pacing['total_scenes'] == 5
    assert 0.0 <= pacing['average_tension'] <= 1.0
    assert isinstance(pacing['recommendations'], list)
    
    print("✅ Pacing analysis working correctly")


def test_pacing_distribution(narrative_sequencer, sample_story):
    """Test pacing speed distribution analysis"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, {})
    
    pacing_dist = continuity.pacing_analysis['pacing_distribution']
    
    # Should have all three pacing types
    assert 'slow' in pacing_dist
    assert 'medium' in pacing_dist
    assert 'fast' in pacing_dist
    
    # Each should have count and percentage
    for pace_data in pacing_dist.values():
        assert 'count' in pace_data
        assert 'percentage' in pace_data
        assert 0 <= pace_data['percentage'] <= 100
    
    print("✅ Pacing distribution calculated correctly")


def test_key_event_extraction(narrative_sequencer):
    """Test extraction of key events from sentences"""
    sentence = "The brave Hero defeats the evil Dragon in an epic Battle"
    
    events = narrative_sequencer._extract_key_events(sentence)
    
    # Should extract capitalized important words
    assert len(events) > 0
    assert any('Hero' in event or 'Dragon' in event or 'Battle' in event for event in events)
    
    print("✅ Key events extracted correctly")


def test_character_extraction_from_sentence(narrative_sequencer):
    """Test character extraction from sentences"""
    sentence = "The seeker meets the teacher at the temple"
    
    characters = narrative_sequencer._extract_characters_from_sentence(sentence)
    
    assert 'seeker' in characters
    assert 'teacher' in characters
    
    print("✅ Characters extracted from sentences")


def test_singleton_pattern():
    """Test that get_narrative_sequencer returns same instance"""
    seq1 = get_narrative_sequencer()
    seq2 = get_narrative_sequencer()
    
    assert seq1 is seq2
    assert id(seq1) == id(seq2)
    
    print("✅ Singleton pattern working")


def test_export_to_json(narrative_sequencer, sample_story, sample_characters, tmp_path):
    """Test JSON export functionality"""
    continuity = narrative_sequencer.analyze_narrative(sample_story, sample_characters)
    
    output_file = tmp_path / "test_narrative.json"
    narrative_sequencer.export_to_json(str(output_file))
    
    assert output_file.exists()
    
    # Load and verify JSON structure
    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert 'story_beats' in data
    assert 'character_arcs' in data
    assert 'dialogue_flows' in data
    assert 'continuity_issues' in data
    assert 'pacing_analysis' in data
    
    print("✅ JSON export working correctly")


def test_empty_story_handling(narrative_sequencer):
    """Test handling of empty story"""
    continuity = narrative_sequencer.analyze_narrative([], {})
    
    assert len(continuity.story_beats) == 0
    assert len(continuity.character_arcs) == 0
    assert len(continuity.dialogue_flows) == 0
    assert "Empty story" in str(continuity.continuity_issues)
    
    print("✅ Empty story handled gracefully")


def test_tension_curve_progression(narrative_sequencer):
    """Test that tension curve follows expected pattern"""
    story = [
        "Life was peaceful in the village",  # Low tension
        "A threat appeared on the horizon",  # Rising
        "The hero prepared for battle",  # Rising
        "An epic final confrontation ensued",  # Peak
        "Peace was restored to the land"  # Falling
    ]
    
    continuity = narrative_sequencer.analyze_narrative(story, {})
    tension_curve = continuity.pacing_analysis['tension_curve']
    
    # Should have 5 tension values
    assert len(tension_curve) == 5
    
    # Find peak
    peak_idx = tension_curve.index(max(tension_curve))
    
    # Peak should be in middle-to-late part of story
    assert peak_idx >= 2  # Not at the very beginning
    
    # Tension after peak should generally decrease
    if peak_idx < len(tension_curve) - 1:
        assert tension_curve[-1] < tension_curve[peak_idx]
    
    print("✅ Tension curve follows expected pattern")


def test_dialogue_type_classification(narrative_sequencer):
    """Test dialogue type classification"""
    exposition = "The teacher explains the ancient wisdom"
    conflict = "They disagrees and argues about the best path forward"
    revelation = "She reveals and discovers the hidden truth"
    
    exp_type = narrative_sequencer._classify_dialogue_type(exposition)
    conf_type = narrative_sequencer._classify_dialogue_type(conflict)
    rev_type = narrative_sequencer._classify_dialogue_type(revelation)
    
    assert exp_type == DialogueType.EXPOSITION
    assert conf_type == DialogueType.CONFLICT
    assert rev_type == DialogueType.REVELATION
    
    print("✅ Dialogue types classified correctly")


def test_growth_indicator_detection(narrative_sequencer):
    """Test detection of character growth indicators"""
    growth_sentence = "The student learned, understood, and mastered the teachings"
    no_growth = "They walked through the forest"
    
    growth_indicators = narrative_sequencer._detect_growth_indicators(growth_sentence)
    no_indicators = narrative_sequencer._detect_growth_indicators(no_growth)
    
    assert len(growth_indicators) >= 2  # Should find 'learned', 'mastered', etc.
    assert len(no_indicators) == 0
    
    print("✅ Growth indicators detected correctly")


def test_integration_with_story_parser(narrative_sequencer):
    """Test integration with story context parser data structure"""
    # Simulate character data from story_context_parser
    characters = {
        'seeker': {
            'gender': 'female',
            'role': 'protagonist',
            'confidence': 0.95,
            'appearances': ['seeker', 'she', 'her']
        },
        'teacher': {
            'gender': 'male',
            'role': 'antagonist',
            'confidence': 0.90,
            'appearances': ['teacher', 'he', 'him']
        }
    }
    
    story = [
        "A young seeker begins her journey",
        "She encounters a wise teacher",
        "They discuss the meaning of existence"
    ]
    
    continuity = narrative_sequencer.analyze_narrative(story, characters)
    
    # Should track both characters
    assert 'seeker' in continuity.character_arcs
    assert 'teacher' in continuity.character_arcs
    
    print("✅ Integration with story parser working")


if __name__ == "__main__":
    print("\n=== TESTING NARRATIVE SEQUENCER V1 ===\n")
    
    # Run tests manually for demo
    from pathlib import Path
    import tempfile
    
    sample_story = [
        "A young seeker embarks on a spiritual journey",
        "She walks through misty forests where ancient sages meditated",
        "The seeker encounters a wise teacher at an old temple",
        "Together they discuss the nature of reality and consciousness",
        "The seeker realizes the truth has been within her all along"
    ]
    
    sample_characters = {
        'seeker': {'gender': 'female', 'role': 'protagonist'},
        'teacher': {'gender': 'male', 'role': 'mentor'}
    }
    
    sequencer = NarrativeSequencerV1()
    
    test_narrative_sequencer_initialization(sequencer)
    test_analyze_narrative(sequencer, sample_story, sample_characters)
    test_story_beat_parsing(sequencer, sample_story)
    test_story_beat_progression(sequencer, sample_story)
    test_character_arc_tracking(sequencer, sample_story, sample_characters)
    test_character_arc_stages(sequencer, sample_story, sample_characters)
    test_dialogue_flow_detection(sequencer)
    test_emotion_detection(sequencer)
    test_continuity_validation(sequencer, sample_story, sample_characters)
    test_continuity_validation_incomplete_story(sequencer)
    test_pacing_analysis(sequencer, sample_story, sample_characters)
    test_pacing_distribution(sequencer, sample_story)
    test_key_event_extraction(sequencer)
    test_character_extraction_from_sentence(sequencer)
    test_singleton_pattern()
    
    # Test export
    with tempfile.TemporaryDirectory() as tmpdir:
        test_export_to_json(sequencer, sample_story, sample_characters, Path(tmpdir))
    
    test_empty_story_handling(sequencer)
    test_tension_curve_progression(sequencer)
    test_dialogue_type_classification(sequencer)
    test_growth_indicator_detection(sequencer)
    test_integration_with_story_parser(sequencer)
    
    print("\n✅ ALL NARRATIVE SEQUENCER TESTS PASSED (20/20)")
