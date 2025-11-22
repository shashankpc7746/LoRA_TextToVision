"""
Test Integration of Emotion Controller with Narrative Sequencer
Demonstrates how emotion controller works with story intelligence

Author: TTV Studio Team
Created: November 17, 2025
"""

import sys
from pathlib import Path

# Add AnimateDiff to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_engine import (
    get_story_context_parser,
    get_scene_memory,
    get_narrative_sequencer,
    get_emotion_controller,
    EmotionType
)


def test_emotion_narrative_integration():
    """Test that emotion controller integrates with narrative sequencer"""
    
    print("\n" + "="*60)
    print("EMOTION CONTROLLER + NARRATIVE SEQUENCER INTEGRATION TEST")
    print("="*60 + "\n")
    
    # Sample story
    story = [
        "A young spiritual seeker begins her journey seeking truth.",
        "She walks through a peaceful forest, contemplating life.",
        "The seeker meets a wise sage who teaches profound wisdom.",
        "In a moment of clarity, she realizes the truth of existence.",
        "Filled with peace, she returns home transformed."
    ]
    
    print("📖 Story:")
    for i, sentence in enumerate(story):
        print(f"   {i+1}. {sentence}")
    print()
    
    # Step 1: Analyze story with story context parser
    print("🧠 Step 1: Analyzing story...")
    story_parser = get_story_context_parser()
    story_analysis = story_parser.analyze_story(story)
    print(f"   ✅ Found {len(story_analysis.characters)} characters")
    
    # Step 2: Build scene graph
    print("\n🎬 Step 2: Building scene graph...")
    scene_memory = get_scene_memory()
    scene_graph = scene_memory.build_scene_graph(story, story_analysis.characters)
    graph_stats = scene_memory.get_graph_stats()
    print(f"   ✅ Scene graph: {graph_stats['total_scenes']} scenes, {graph_stats['total_entities']} entities")
    
    # Step 3: Analyze narrative structure
    print("\n📊 Step 3: Analyzing narrative structure...")
    narrative_seq = get_narrative_sequencer()
    narrative_analysis = narrative_seq.analyze_narrative(story, story_analysis.characters)
    print(f"   ✅ Story beats parsed: {len(narrative_analysis.story_beats)} beats")
    print(f"   ✅ Character arcs tracked: {len(narrative_analysis.character_arcs)} characters")
    
    # Print story beats
    print("\n   Story Beat Progression:")
    for i, beat in enumerate(narrative_analysis.story_beats):
        print(f"      Scene {i}: {beat.beat_type.value} (tension: {beat.tension_level:.2f}, pace: {beat.pacing_speed})")
    
    # Step 4: Map emotions based on narrative beats
    print("\n🎭 Step 4: Mapping emotions based on narrative structure...")
    emotion_controller = get_emotion_controller()
    emotion_controller.reset()
    
    # Get the main character
    main_character = list(story_analysis.characters.values())[0].name
    print(f"   Main character: {main_character}")
    
    # Map emotions based on story beats
    emotion_mapping = {
        "setup": (EmotionType.NEUTRAL, 0.5),
        "inciting_incident": (EmotionType.CONTEMPLATION, 0.6),
        "rising_action": (EmotionType.DETERMINATION, 0.7),
        "climax": (EmotionType.AWE, 0.9),
        "falling_action": (EmotionType.PEACE, 0.8),
        "resolution": (EmotionType.PEACE, 0.9)
    }
    
    # Set emotions based on story beats
    for i, beat in enumerate(narrative_analysis.story_beats):
        beat_type = beat.beat_type.value
        if beat_type in emotion_mapping:
            emotion, intensity = emotion_mapping[beat_type]
            emotion_controller.set_emotion(
                character_name=main_character,
                emotion=emotion,
                intensity=intensity,
                scene_index=i
            )
            print(f"   Scene {i}: {emotion.value} (intensity: {intensity:.1f})")
    
    # Step 5: Add emotion transitions between scenes
    print("\n🌊 Step 5: Creating smooth emotional transitions...")
    transitions_created = 0
    for i in range(len(story) - 1):
        current_emotion = emotion_controller.get_current_emotion(main_character, i)
        next_emotion = emotion_controller.get_current_emotion(main_character, i + 1)
        
        if current_emotion and next_emotion and current_emotion.emotion != next_emotion.emotion:
            transition = emotion_controller.transition_emotion(
                character_name=main_character,
                from_scene=i,
                to_scene=i + 1,
                to_emotion=next_emotion.emotion,
                to_intensity=next_emotion.intensity,
                transition_frames=15
            )
            transitions_created += 1
            print(f"   Transition {i}→{i+1}: {current_emotion.emotion.value} → {next_emotion.emotion.value}")
    
    print(f"   ✅ Created {transitions_created} smooth transitions")
    
    # Step 6: Add micro-expressions at key moments
    print("\n✨ Step 6: Adding micro-expressions at key moments...")
    # Add surprise at the climax (scene 3)
    emotion_controller.schedule_micro_expression(
        character_name=main_character,
        emotion=EmotionType.SURPRISE,
        intensity=0.7,
        scene_index=3,
        start_frame=15,
        duration_frames=10
    )
    print(f"   Scene 3: Added SURPRISE micro-expression at climax")
    
    # Add confusion during contemplation (scene 1)
    emotion_controller.schedule_micro_expression(
        character_name=main_character,
        emotion=EmotionType.CONFUSION,
        intensity=0.4,
        scene_index=1,
        start_frame=20,
        duration_frames=8
    )
    print(f"   Scene 1: Added CONFUSION micro-expression during contemplation")
    
    # Step 7: Calculate motion parameters for each scene
    print("\n🎯 Step 7: Calculating motion parameters based on emotions...")
    for i in range(len(story)):
        motion_params = emotion_controller.calculate_emotional_motion(main_character, i)
        if motion_params:
            print(f"   Scene {i}:")
            print(f"      Speed: {motion_params.speed_multiplier:.2f}x")
            print(f"      Gesture amplitude: {motion_params.gesture_amplitude:.2f}")
            print(f"      Body tension: {motion_params.body_tension:.2f}")
    
    # Step 8: Validate emotional arc
    print("\n✅ Step 8: Validating emotional arc...")
    validation_report = emotion_controller.validate_emotional_arc(main_character)
    print(f"   Arc valid: {validation_report['valid']}")
    print(f"   Total emotions: {validation_report['total_emotions']}")
    print(f"   Unique emotions: {validation_report['unique_emotions']}")
    print(f"   Emotion variety: {', '.join([e.value for e in validation_report['emotion_variety']])}")
    
    if validation_report['issues']:
        print(f"\n   ⚠️  Issues found:")
        for issue in validation_report['issues']:
            print(f"      - {issue}")
    
    if validation_report['recommendations']:
        print(f"\n   💡 Recommendations:")
        for rec in validation_report['recommendations']:
            print(f"      - {rec}")
    
    # Step 9: Export for visualization
    print("\n💾 Step 9: Exporting results...")
    output_path = emotion_controller.export_to_json("emotion_narrative_integration.json")
    print(f"   ✅ Exported to: {output_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE! ✅")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Story analyzed: 5 scenes")
    print(f"   • Narrative beats: {len(narrative_analysis.story_beats)} beats")
    print(f"   • Emotions mapped: {validation_report['total_emotions']} states")
    print(f"   • Transitions created: {transitions_created} smooth transitions")
    print(f"   • Micro-expressions: 2 subtle expressions")
    print(f"   • Motion parameters: Calculated for all scenes")
    print(f"   • Emotional arc: {'Valid' if validation_report['valid'] else 'Needs improvement'}")
    print()


if __name__ == "__main__":
    test_emotion_narrative_integration()
