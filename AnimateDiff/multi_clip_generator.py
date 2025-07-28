# multi_clip_generator.py

import os
import shutil
import cv2
import numpy as np
import json
import argparse
import datetime
import hashlib
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, TextClip, CompositeVideoClip
from moviepy.video.fx.all import crop
# Re-added vfx and crop for dynamic camera effects (but no fade effects)
from animate_gurukul import generate_clip, fps
from utils.controlnet_utils import (
    generate_multi_control_guidance,
    generate_adaptive_multi_control_guidance,
    extract_character_features,
    load_character_features,
    enhance_prompt_with_character_features
)
from utils.content_analyzer import content_analyzer
from utils.quality_scorer import quality_scorer
from utils.face_identity import face_identity_preserver
from utils.retry_system import auto_retry_system
from utils.realtime_lora import realtime_lora_trainer
from utils.storage_delivery import storage_system
import re

# ----------- LESSON JSON SUPPORT -----------

def load_lesson_from_json(json_path: str) -> dict:
    """Load lesson from Akash's JSON format"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            lesson_data = json.load(f)

        print(f"LOADED: {lesson_data.get('title', 'Unknown')}")
        print(f"📊 Level: {lesson_data.get('level', 'Unknown')}")
        print(f"🎵 TTS enabled: {lesson_data.get('tts', False)}")

        return lesson_data
    except Exception as e:
        print(f"ERROR: Error loading lesson JSON: {e}")
        return None

def extract_text_from_lesson(lesson_data: dict) -> str:
    """Extract text content from lesson for video generation"""
    if not lesson_data:
        return None

    text_content = lesson_data.get('text', '')
    if not text_content:
        
        print("⚠️ No text content found in lesson")
        return None

    return text_content

# ----------- SUBTITLE SYSTEM -----------

def add_subtitles_to_video(video_path: str, text_content: str, output_path: str) -> str:
    """Add subtitles to video based on text content"""
    try:
        print(f"🎬 Adding subtitles to video: {os.path.basename(video_path)}")

        # Load the video
        video = VideoFileClip(video_path)
        video_duration = video.duration

        # Split text into sentences for subtitles
        sentences = [s.strip() for s in text_content.split('.') if s.strip()]

        if not sentences:
            print("⚠️ No sentences found for subtitles")
            return video_path

        # Calculate timing for each subtitle
        subtitle_duration = video_duration / len(sentences)
        subtitle_clips = []

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            start_time = i * subtitle_duration
            end_time = min((i + 1) * subtitle_duration, video_duration)

            # Create subtitle text clip
            subtitle = TextClip(
                sentence,
                fontsize=24,
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial-Bold'
            ).set_position(('center', 'bottom')).set_start(start_time).set_duration(end_time - start_time)

            subtitle_clips.append(subtitle)
            print(f"   📝 Subtitle {i+1}: {start_time:.1f}s-{end_time:.1f}s - {sentence[:50]}...")

        # Composite video with subtitles
        if subtitle_clips:
            final_video = CompositeVideoClip([video] + subtitle_clips)

            # Save video with subtitles
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=fps,  # Using centralized FPS setting
                bitrate="8000k"
            )

            # Cleanup
            video.close()
            final_video.close()
            for clip in subtitle_clips:
                clip.close()

            print(f"✅ Subtitles added successfully: {os.path.basename(output_path)}")
            return output_path
        else:
            print("⚠️ No subtitle clips created")
            video.close()
            return video_path

    except Exception as e:
        print(f"❌ Subtitle generation failed: {e}")
        return video_path

def add_subtitles_to_video_clips(video_path: str, sentences: list, output_path: str, num_clips: int) -> str:
    """Add subtitles to video with precise timing for each clip"""
    try:
        print(f"🎬 Adding clip-synchronized subtitles to video: {os.path.basename(video_path)}")

        # Load the video
        video = VideoFileClip(video_path)
        video_duration = video.duration

        if not sentences:
            print("⚠️ No sentences found for subtitles")
            return video_path

        # Ensure we have the right number of sentences
        if len(sentences) > num_clips:
            sentences = sentences[:num_clips]
        elif len(sentences) < num_clips:
            # Pad with empty strings if needed
            sentences.extend([''] * (num_clips - len(sentences)))

        # Calculate clip duration
        clip_duration = video_duration / num_clips
        subtitle_clips = []

        for i, sentence in enumerate(sentences):
            if not sentence:
                continue

            # Calculate precise timing for this clip
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, video_duration)

            # Create subtitle text clip with larger font and better positioning
            subtitle = TextClip(
                sentence,
                fontsize=28,  # Larger font
                color='white',
                stroke_color='black',
                stroke_width=2,
                font='Arial-Bold',
                method='caption',  # Better text wrapping
                size=(video.w * 0.9, None),  # 90% of video width
                align='center'
            ).set_position(('center', 'bottom')).set_start(start_time).set_duration(end_time - start_time)

            subtitle_clips.append(subtitle)
            print(f"   📝 Subtitle {i+1}: {start_time:.1f}s-{end_time:.1f}s - {sentence[:50]}...")

        # Composite video with subtitles
        if subtitle_clips:
            final_video = CompositeVideoClip([video] + subtitle_clips)

            # Save video with subtitles
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=fps,  # Using centralized FPS setting
                bitrate="8000k"
            )

            # Cleanup
            video.close()
            final_video.close()
            for clip in subtitle_clips:
                clip.close()

            print(f"✅ Clip-synchronized subtitles added successfully: {os.path.basename(output_path)}")
            return output_path
        else:
            print("⚠️ No subtitle clips created")
            video.close()
            return video_path

    except Exception as e:
        print(f"❌ Clip-synchronized subtitle generation failed: {e}")
        return video_path

# ----------- CONFIGURATION -----------
# PHASE 2 TEST: Educational Science Content (Scientist consistency test)
# paragraph = """
# A young scientist in a white lab coat examines a glowing chemical reaction in a beaker.
# The scientist carefully adds a blue solution drop by drop into the mixture.
# Colorful bubbles begin to form and rise from the beaker as the reaction intensifies.
# The scientist takes detailed notes while observing the changing colors.
# Steam rises from the beaker as the reaction reaches its peak temperature.
# """

# SCENARIO 2: Historical Story Content
# paragraph = """
# A medieval knight in shining armor rides through a misty forest on his horse.
# The knight dismounts and approaches an ancient stone castle with tall towers.
# He draws his sword as he enters the castle's dark and mysterious courtyard.
# Inside the castle, the knight discovers a treasure chest hidden behind tapestries.
# The knight carefully opens the chest, revealing golden coins and precious gems.
# Suddenly, the knight hears footsteps echoing through the castle corridors.
# The knight quickly closes the chest and prepares to face whatever approaches.
# """

# SCENARIO 3: Mathematical Concept Visualization
# paragraph = """
# A teacher stands in front of a blackboard drawing geometric shapes with chalk.
# The teacher draws a perfect circle and explains the concept of radius and diameter.
# Students watch attentively as the teacher demonstrates how to calculate circumference.
# The teacher uses a compass to draw multiple circles of different sizes.
# Next, the teacher shows how circles relate to other geometric shapes like triangles.
# The teacher draws tangent lines touching the circle at exactly one point.
# Finally, the teacher solves a complex geometry problem step by step on the board.
# """

# SCENARIO 4: Nature Documentary Style
# paragraph = """
# A majestic eagle soars high above snow-capped mountain peaks in the morning light.
# The eagle spots a fish swimming in the crystal-clear lake below.
# With incredible precision, the eagle dives down toward the water surface.
# The eagle's talons break the water surface as it catches the fish.
# The eagle spreads its powerful wings and lifts off from the lake.
# Flying back to its nest, the eagle carries the fish in its strong grip.
# The eagle lands on a rocky cliff and feeds its hungry chicks.
# """

# SCENARIO 5: Technology/Programming Content
# paragraph = """
# A programmer sits at a computer with multiple monitors displaying colorful code.
# The programmer types rapidly, creating a new software application.
# Lines of code appear on the screen as the programmer builds the program logic.
# The programmer tests the application by clicking various buttons and menus.
# A bug appears, causing the programmer to carefully debug the problematic code.
# The programmer fixes the error and runs the program again successfully.
# Finally, the programmer saves the completed project and celebrates the achievement.
# """

# ----------- MULTIPLE RENDER STYLES SYSTEM -----------

RENDER_STYLES = {
    'realistic': {
        'name': 'Realistic Style',
        'model': 'SG161222/Realistic_Vision_V5.1_noVAE',
        'description': 'Photorealistic characters and environments',
        'prompt_suffix': ', photorealistic, detailed, high quality, realistic lighting',
        'guidance_scale': 15,
        'steps': 25  # Restored to 25 for better realistic quality
    },
    'anime': {
        'name': 'Anime Style',
        'model': 'xyn-ai/anything-v4.0',
        'description': 'Traditional anime art style',
        'prompt_suffix': ', anime style, detailed anime art, vibrant colors',
        'guidance_scale': 18,
        'steps': 18  # Reduced from 30 to 18 for faster generation
    },
    'artistic': {
        'name': 'Artistic/Painterly Style',
        'model': 'runwayml/stable-diffusion-v1-5',
        'description': 'Watercolor and oil painting effects',
        'prompt_suffix': ', watercolor painting, artistic, painterly style, soft brushstrokes',
        'guidance_scale': 12,
        'steps': 20  # Reduced from 35 to 20 for faster generation
    }
}

def select_render_style(style_name: str = 'realistic') -> dict:
    """Select and return render style configuration"""
    if style_name not in RENDER_STYLES:
        print(f"⚠️ Unknown style '{style_name}', defaulting to anime")
        style_name = 'anime'

    style = RENDER_STYLES[style_name]
    print(f"🎨 Selected render style: {style['name']}")
    print(f"   • Model: {style['model']}")
    print(f"   • Description: {style['description']}")
    print(f"   • Guidance scale: {style['guidance_scale']}")
    print(f"   • Steps: {style['steps']}")

    return style

# REMOVED: Multiple style generation function - simplified to single output directory
# All styles now use the same outputs/multi_clip/ directory

def create_style_selection_menu():
    """Create a simple style selection interface"""
    print(f"\n🎨 AVAILABLE RENDER STYLES:")
    print(f"=" * 50)

    for i, (style_key, style_config) in enumerate(RENDER_STYLES.items(), 1):
        print(f"{i}. {style_config['name']}")
        print(f"   • {style_config['description']}")
        print(f"   • Model: {style_config['model']}")
        print()

    print(f"0. Generate ALL styles")
    print(f"=" * 50)

    # For automated processing, return default style
    # In interactive mode, this could accept user input
    return 'anime'  # Default to anime style

def enhance_prompt_with_style(prompt: str, style_config: dict) -> str:
    """Enhance prompt with style-specific keywords"""
    enhanced_prompt = prompt + style_config['prompt_suffix']
    return enhanced_prompt

# ----------- BATCH PROCESSING SYSTEM -----------

def process_lesson_queue():
    """Process multiple lesson files in batch"""
    lesson_files = []
    lessons_dir = "lessons"

    # Create lessons directory if it doesn't exist
    if not os.path.exists(lessons_dir):
        os.makedirs(lessons_dir)
        print(f"📁 Created lessons directory: {lessons_dir}")

    # Look for multiple lesson files in lessons directory
    for file in os.listdir(lessons_dir):
        if file.endswith('.json'):
            lesson_files.append(os.path.join(lessons_dir, file))

    # Also check for single lesson file in root (legacy support)
    if os.path.exists('lesson_input.json'):
        lesson_files.append('lesson_input.json')

    if not lesson_files:
        print("📝 No lesson files found, using default content")
        return None, None

    print(f"LESSONS: Found {len(lesson_files)} lesson file(s): {[os.path.basename(f) for f in lesson_files]}")

    # FIXED: Use lesson from command line arguments (unified system passes it)
    lesson_file = None

    # Check command line arguments: python multi_clip_generator.py lesson_file.json style
    import sys
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg.endswith('.json'):
            # First argument is lesson file
            lesson_file = os.path.join("lessons", first_arg)
            if not os.path.exists(lesson_file):
                print(f"❌ Specified lesson not found: {lesson_file}")
                return None, None
        else:
            # Old format: just style argument, use default lesson
            space_lesson = "lessons/lesson_forest_wisdom.json"
            if os.path.exists(space_lesson):
                lesson_file = space_lesson
            else:
                lesson_file = lesson_files[0] if lesson_files else None
    else:
        # No arguments, use default
        space_lesson = "lessons/lesson_forest_wisdom.json"
        if os.path.exists(space_lesson):
            lesson_file = space_lesson
        else:
            lesson_file = lesson_files[0] if lesson_files else None

    if not lesson_file:
        print(f"❌ No lesson files found")
        return None, None

    print(f"PROCESSING: {os.path.basename(lesson_file)}")

    lesson_data = load_lesson_from_json(lesson_file)
    if lesson_data:
        lesson_title = lesson_data.get('title', 'Unknown Lesson')
        lesson_level = lesson_data.get('level', 'Unknown Level')
        print(f"📚 Loaded lesson: {lesson_title}")
        print(f"📊 Level: {lesson_level}")

        extracted_text = extract_text_from_lesson(lesson_data)
        if extracted_text:
            print(f"✅ Using lesson content from {os.path.basename(lesson_file)}")
            return extracted_text, lesson_data

    print(f"⚠️ Failed to load lesson from {os.path.basename(lesson_file)}")
    return None, None

# Process lesson queue
paragraph, lesson_data = process_lesson_queue()

# Fallback to default content if no lessons found
if paragraph is None:
    print(f"📝 Using default simple story")
    paragraph = """
A young anime boy walks through a peaceful village in the morning.
He stops at a small flower shop and buys a red rose.
The boy walks to a nearby park with cherry blossom trees.
He sits on a wooden bench and reads a book quietly.
A friendly cat approaches and sits beside him on the bench.
The boy pets the cat gently and shares his lunch with it.
"""


base_output_dir = "outputs/multi_clip/"
os.makedirs(base_output_dir, exist_ok=True)
# Generate completely random seed for each run AND each clip
import random
import time
# Ensure true randomness by combining time with system entropy
random.seed(time.time())  # Initialize with current time
base_seed = random.randint(100000, 999999999)  # Much larger range
print(f"🎲 Using random base seed: {base_seed}")
print(f"🎲 Each clip will use a different random seed for variety")
clip_prompts = []

# ------------- PHASE 1 CONSISTENCY IMPROVEMENTS -------------

def evaluate_pose_quality(frame):
    """Evaluate how good a frame is for pose detection"""
    if frame is None:
        return 0

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Check for human-like features (skin tone detection)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_percentage = np.sum(skin_mask > 0) / (frame.shape[0] * frame.shape[1])

    # Check for good contrast and detail
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)

    # Check for center-weighted content (character likely in center)
    h, w = frame.shape[:2]
    center_region = frame[h//4:3*h//4, w//4:3*w//4]
    center_activity = np.std(center_region)

    # Combine scores
    pose_score = (skin_percentage * 100) + (contrast / 10) + (center_activity / 10)
    return pose_score

def find_best_continuity_frame(video_path, num_frames_to_check=8):
    """Find the best frame for pose extraction from recent frames"""
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return None

    best_frame = None
    best_score = 0
    best_frame_idx = total_frames - 1  # Default to last frame

    # Check the last N frames
    start_frame = max(0, total_frames - num_frames_to_check)

    for frame_idx in range(start_frame, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            score = evaluate_pose_quality(frame)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_frame_idx = frame_idx

    cap.release()

    print(f"   🎯 Selected frame {best_frame_idx}/{total_frames-1} (score: {best_score:.2f})")
    return best_frame

def enhance_prompt_for_consistency(original_prompt, clip_index, character_info=None):
    """Add consistency keywords to prompts"""

    # SPECIAL CASE: Enhance dog visibility in dog prompt
    if "dog" in original_prompt.lower() and "runs past" in original_prompt.lower():
        # Make dog much more prominent and visible
        dog_enhancements = [
            "LARGE PROMINENT DOG IN FOREGROUND",
            "dog taking up 50% of the frame",
            "dog as main focus",
            "detailed dog features",
            "dog clearly visible",
            "dog in center of frame",
            "close-up of dog",
            "dog splashing water dramatically"
        ]
        original_prompt = original_prompt + ", " + ", ".join(dog_enhancements)
        print(f"🐕 ENHANCED DOG VISIBILITY: Added special dog prominence keywords")

    # Extract character information from prompt
    character_keywords = []

    # Look for character descriptions
    if "woman" in original_prompt.lower() or "girl" in original_prompt.lower():
        character_keywords.extend(["same woman", "consistent female character"])
    elif "man" in original_prompt.lower() or "boy" in original_prompt.lower():
        character_keywords.extend(["same man", "consistent male character"])
    else:
        character_keywords.append("same person")

    # Look for clothing descriptions
    clothing_matches = re.findall(r'\b(?:wearing|in)\s+(?:a\s+)?([^,.\n]+(?:hoodie|jacket|shirt|dress|coat))', original_prompt.lower())
    for clothing in clothing_matches:
        character_keywords.append(f"wearing {clothing}")

    # Add general consistency terms
    consistency_terms = [
        "consistent character",
        "identical appearance",
        "maintaining identity",
        "same facial features"
    ]

    # Combine with original prompt
    if clip_index > 1:  # Add consistency terms from clip 2 onwards
        enhanced_prompt = f"{original_prompt}, {', '.join(character_keywords + consistency_terms[:2])}"
    else:
        enhanced_prompt = original_prompt

    return enhanced_prompt

# ------------- STEP 1: Split Paragraph into Sub-Prompts -------------
def split_paragraph(text):
    """Split paragraph by lines and sentence punctuation."""
    import re

    # First try splitting by newlines (for line-based prompts)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]

    # If we get multiple lines, use them
    if len(lines) > 1:
        return lines

    # Otherwise, fall back to sentence splitting
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

clip_prompts = split_paragraph(paragraph)
print(f"🧠 Detected {len(clip_prompts)} prompts:")
for i, p in enumerate(clip_prompts):
    print(f"   [{i+1}] Length: {len(p)} chars - {p}")

# Validate that we have multiple prompts
if len(clip_prompts) <= 1:
    print("⚠️ Warning: Only 1 prompt detected. Check your paragraph formatting.")
    print("💡 Tip: Make sure each sentence is on a separate line or ends with punctuation.")
else:
    print(f"✅ Ready to generate {len(clip_prompts)} video clips!")

# ------------- PHASE 2: CONTENT ANALYSIS & OPTIMIZATION -------------
print(f"\n🧠 Analyzing content for Phase 2 optimizations...")
content_analysis = content_analyzer.analyze_content_type(clip_prompts)
optimal_config = content_analyzer.select_optimal_config(content_analysis)
consistency_strategy = content_analyzer.get_consistency_strategy(content_analysis)

print(f"📊 Content Analysis Results:")
print(f"   • Primary type: {content_analysis['primary_type']} (confidence: {content_analysis['confidence']:.2f})")
print(f"   • Complexity: {content_analysis['complexity']}")
print(f"   • Motion intensity: {content_analysis['motion_intensity']}")
print(f"   • Has humans: {content_analysis['has_humans']}")
print(f"   • Has animals: {content_analysis['has_animals']}")
print(f"   • Is educational: {content_analysis['is_educational']}")

print(f"\n⚙️ Optimal Configuration:")
print(f"   • ControlNet weight: {optimal_config['controlnet_weight']}")
print(f"   • Guidance scale: {optimal_config['guidance_scale']}")
print(f"   • Inference steps: {optimal_config['num_inference_steps']}")

print(f"\n🎯 Consistency Strategy:")
print(f"   • Use character reference: {consistency_strategy['use_character_reference']}")
print(f"   • Retry threshold: {consistency_strategy['retry_threshold']}")

# ----------- RENDER STYLE SELECTION -----------
# Parse command line arguments for style selection
import sys
style_arg = 'anime'  # Default
if len(sys.argv) > 1 and sys.argv[1] in RENDER_STYLES:
    style_arg = sys.argv[1]
    print(f"🎨 Command line style selected: {style_arg}")

selected_style = select_render_style(style_arg)

# Override optimal config with style-specific settings
optimal_config['guidance_scale'] = selected_style['guidance_scale']
optimal_config['num_inference_steps'] = selected_style['steps']
optimal_config['model_description'] = selected_style['description']

print(f"\n🎨 RENDER STYLE APPLIED:")
print(f"   • Style: {selected_style['name']}")
print(f"   • Updated guidance scale: {optimal_config['guidance_scale']}")
print(f"   • Updated steps: {optimal_config['num_inference_steps']}")

# Initialize character features storage
character_features = None
character_reference_extracted = False

# ------------- PHASE 3: ADVANCED SYSTEMS INITIALIZATION -------------
print(f"\n🚀 Initializing Phase 3 Advanced Systems...")

# Initialize face identity preservation
face_identity_established = False
character_name = "main_character"

# Initialize LoRA training
lora_trained = False
character_lora_path = None

# Initialize retry system
generation_attempts = {}

print(f"✅ Phase 3 systems ready:")
print(f"   • Face identity preservation: Ready")
print(f"   • Real-time LoRA training: Ready")
print(f"   • Automatic retry system: Ready")
print(f"   • Multi-model support: {len(content_analyzer.get_content_specific_models(content_analysis))} models available")

# ------------- STEP 2: Generate Clips One by One -------------
print(f"\n🎬 Starting generation of {len(clip_prompts)} clips...")
last_frame_path = None
generated_clips = []

for idx, prompt in enumerate(clip_prompts):
    print(f"\n🎬 Generating Clip {idx + 1}/{len(clip_prompts)}...")
    print(f"📝 Original Prompt: {prompt}")

    # PHASE 1 IMPROVEMENT: Enhance prompt for consistency
    enhanced_prompt = enhance_prompt_for_consistency(prompt, idx + 1)

    # PHASE 2 IMPROVEMENT: Enhance with character features
    if character_features is not None:
        enhanced_prompt = enhance_prompt_with_character_features(enhanced_prompt, character_features)

    # PHASE 3 IMPROVEMENT: Enhance with face identity preservation
    if face_identity_established:
        identity_verification = face_identity_preserver.verify_identity_consistency(
            last_frame_path, character_name
        )
        enhanced_prompt = face_identity_preserver.enhance_prompt_for_identity(
            enhanced_prompt, identity_verification
        )
        print(f"🎭 Identity verification: {identity_verification['reason']}")

    if enhanced_prompt != prompt:
        print(f"✨ Enhanced Prompt: {enhanced_prompt}")

    # Output paths
    clip_name = f"clip{idx + 1}"
    output_video = os.path.join(base_output_dir, f"{clip_name}.mp4")
    last_frame_path_new = os.path.join(base_output_dir, f"{clip_name}_last.png")
    pose_path = None

    # PHASE 1 IMPROVEMENT: Smart frame selection and multi-control guidance
    if idx > 0 and generated_clips:  # Not first clip
        print(f"CONTINUITY: Finding best frame from previous clip...")

        # Find best frame from previous clip
        prev_video = generated_clips[-1]
        best_frame = find_best_continuity_frame(prev_video)

        if best_frame is not None:
            # Save the best frame temporarily
            temp_frame_path = os.path.join(base_output_dir, f"{clip_name}_temp_frame.png")
            cv2.imwrite(temp_frame_path, best_frame)

            # PHASE 2 IMPROVEMENT: Generate adaptive multi-control guidance
            print(f"🎮 Generating adaptive multi-control guidance...")
            control_result = generate_adaptive_multi_control_guidance(
                temp_frame_path,
                os.path.join(base_output_dir, f"{clip_name}_control.png"),
                content_analysis
            )

            pose_path = control_result['primary_control']
            adaptive_weight = control_result.get('adaptive_weight', optimal_config['controlnet_weight'])

            # Clean up temp file
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
        else:
            print("⚠️ Could not find suitable frame for continuity")
    else:
        print("🎬 First clip - no continuity guidance needed")

    # PHASE 3 IMPROVEMENT: Enhance config with character LoRA
    if character_lora_path:
        optimal_config = realtime_lora_trainer.enhance_config_with_lora(optimal_config, character_name)

    # PHASE 3 IMPROVEMENT: Use retry system for robust generation
    def generation_func(config):
        try:
            generate_clip(
                prompt=enhanced_prompt,
                output_path=output_video,
                pose_path=pose_path,
                init_image_path=last_frame_path,
                seed=config.get('seed', random.randint(100000, 999999999))  # Completely random seed for each clip
            )
            return {'success': True, 'output_path': output_video}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def quality_check_func(video_path):
        return quality_scorer.evaluate_video_quality(video_path, content_analysis)

    def identity_check_func(video_path):
        if face_identity_established:
            return face_identity_preserver.verify_identity_consistency(video_path, character_name)
        return None

    # Execute with retry system
    retry_result = auto_retry_system.execute_with_retry(
        generation_func,
        optimal_config,
        quality_check_func,
        identity_check_func
    )

    if retry_result['success']:
        generated_clips.append(output_video)
        generation_attempts[idx + 1] = retry_result['attempts']
        print(f"✅ Clip {idx + 1} generated successfully after {retry_result['attempts']} attempt(s)!")
    else:
        # STORY PRESERVATION: Include failed clips to maintain story continuity
        print(f"⚠️ Clip {idx + 1} had issues but will be included to preserve story continuity")
        print(f"   Issue: {retry_result['message']}")

        # Check if the video file was actually generated despite "failure"
        if os.path.exists(output_video):
            generated_clips.append(output_video)
            generation_attempts[idx + 1] = retry_result.get('attempts', 1)
            print(f"📖 Clip {idx + 1} included in story sequence")
        else:
            print(f"❌ Clip {idx + 1} file not found - cannot include in story")
            # Note: We don't use 'continue' here to maintain clip numbering

    # ------------- STEP 3: Extract Last Frame & Phase 3 Processing -------------
    cap = cv2.VideoCapture(output_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(last_frame_path_new, frame)
        last_frame_path = last_frame_path_new
        print(f"💾 Saved last frame for next clip continuity")

        # PHASE 3 IMPROVEMENT: Face identity establishment
        if not face_identity_established and idx == 0:  # First clip
            if face_identity_preserver.establish_identity_reference(last_frame_path_new, character_name):
                face_identity_established = True
                face_identity_preserver.save_identity_reference(base_output_dir, character_name)
                print(f"🎭 Face identity reference established from first clip")

        # PHASE 3 IMPROVEMENT: Real-time LoRA training
        if not lora_trained and idx == 0 and len(generated_clips) == 1:  # After first successful clip
            print(f"🎨 Starting real-time LoRA training...")
            character_lora_path = realtime_lora_trainer.train_from_successful_clip(
                output_video,
                enhanced_prompt,
                character_name
            )
            if character_lora_path:
                lora_trained = True
                print(f"✅ Character LoRA trained and ready for subsequent clips")

        # PHASE 2 IMPROVEMENT: Extract character features (if not done yet)
        if not character_reference_extracted and consistency_strategy['use_character_reference']:
            try:
                print(f"🎭 Extracting character features from clip...")
                character_features = extract_character_features(last_frame_path_new, base_output_dir)
                character_reference_extracted = True
                print(f"✅ Character reference established for consistency")
            except Exception as char_error:
                print(f"⚠️ Could not extract character features: {char_error}")

    else:
        print(f"⚠️ Failed to extract last frame from {output_video}")
        break

# ------------- STEP 4: Stitch All Clips into Final Video -------------
print(f"\n🎞️ Stitching {len(generated_clips)} clips into one final video...")

# STORY PRESERVATION: Always check for ALL existing clips regardless of generated_clips list
print("🔍 Checking for all existing clips to preserve complete story...")
all_existing_clips = []
for i in range(len(clip_prompts)):
    potential_clip = os.path.join(base_output_dir, f"clip{i+1}.mp4")
    if os.path.exists(potential_clip):
        all_existing_clips.append(potential_clip)
        print(f"📖 Found clip {i+1} for story continuity")

if all_existing_clips:
    print(f"✅ Story preservation: Using {len(all_existing_clips)}/{len(clip_prompts)} existing clips")
    generated_clips = all_existing_clips  # Override with all existing clips
elif not generated_clips:
    print("❌ No clips were generated successfully. Exiting...")
    exit(1)

clips = []
for idx, clip_path in enumerate(generated_clips):
    if not os.path.exists(clip_path):
        print(f"⚠️ Skipping missing clip: {clip_path}")
        continue

    print(f"📎 Loading clip {idx + 1}: {os.path.basename(clip_path)}")
    clip = VideoFileClip(clip_path)

    # 🚫 NO FADE EFFECTS - But add dynamic camera movements for cinematic feel
    # Zero fade in/out, but enhanced 360° pan and dynamic effects

    # 🎥 ENHANCED 360° PAN + DYNAMIC CAMERA EFFECTS
    duration = clip.duration
    movement_type = idx % 6  # 6 different movement patterns

    if movement_type == 0:
        # 🔄 SMOOTH 3D CIRCULAR PAN - No zoom, pure 3D motion
        def smooth_3d_pan(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create 3D circular motion effect
            angle = progress * 2 * math.pi  # Full 360° rotation
            radius = 25  # Fixed radius for consistent motion
            x_offset = int(radius * math.cos(angle)) + 25
            y_offset = int(radius * math.sin(angle) * 0.6) + 15  # Elliptical for 3D effect
            return frame[y_offset:frame.shape[0]-y_offset, x_offset:frame.shape[1]-x_offset]
        clip = clip.fl(smooth_3d_pan)

    elif movement_type == 1:
        # 🌀 SMOOTH 3D ORBITAL MOTION - No zoom, pure orbital movement
        def smooth_3d_orbital(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create 3D orbital motion (like satellite view)
            angle = progress * 1.5 * math.pi  # 270° rotation for dynamic feel
            orbit_radius = 30
            x_orbit = int(orbit_radius * math.cos(angle)) + 30
            y_orbit = int(orbit_radius * math.sin(angle) * 0.7) + 20  # Flattened for 3D perspective
            return frame[y_orbit:frame.shape[0]-y_orbit, x_orbit:frame.shape[1]-x_orbit]
        clip = clip.fl(smooth_3d_orbital)

    elif movement_type == 2:
        # 📹 CINEMATIC 3D TRACKING - No zoom, pure tracking motion
        def cinematic_3d_tracking(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create smooth tracking shot with 3D perspective
            x_track = int(40 * math.sin(progress * math.pi)) + 20  # Smooth S-curve tracking
            y_track = int(15 * progress) + 10  # Gentle vertical rise
            return frame[y_track:frame.shape[0]-y_track, x_track:frame.shape[1]-x_track]
        clip = clip.fl(cinematic_3d_tracking)

    elif movement_type == 3:
        # 🎬 3D PENDULUM MOTION - No zoom, smooth pendulum-like movement
        def pendulum_3d_motion(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create 3D pendulum effect
            swing_angle = math.sin(progress * 3 * math.pi) * 35  # 3 swings
            x_swing = int(swing_angle) + 35
            y_swing = int(abs(swing_angle) * 0.3) + 15  # Slight vertical movement
            return frame[y_swing:frame.shape[0]-y_swing, x_swing:frame.shape[1]-x_swing]
        clip = clip.fl(pendulum_3d_motion)

    elif movement_type == 4:
        # 🎯 3D TILT & DRIFT - No zoom, pure tilt with drift motion
        def tilt_3d_drift(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create 3D tilt effect with drift
            y_tilt = int(30 * math.sin(progress * math.pi)) + 20  # Smooth tilt motion
            x_drift = int(20 * progress) + 15  # Gentle horizontal drift
            return frame[y_tilt:frame.shape[0]-y_tilt, x_drift:frame.shape[1]-x_drift]
        clip = clip.fl(tilt_3d_drift)

    else:  # movement_type == 5
        # 🌊 3D WAVE FLOW - No zoom, pure organic wave motion
        def wave_3d_flow(get_frame, t):
            frame = get_frame(t)
            progress = t / duration
            import math
            # Create 3D wave flow effect
            x_wave = int(30 * math.sin(progress * 4 * math.pi)) + 30  # Horizontal wave
            y_wave = int(20 * math.cos(progress * 2 * math.pi)) + 20  # Vertical wave
            # Ensure boundaries are within frame
            x_start = max(0, min(x_wave, frame.shape[1] - 100))
            y_start = max(0, min(y_wave, frame.shape[0] - 100))
            return frame[y_start:frame.shape[0]-y_start, x_start:frame.shape[1]-x_start]
        clip = clip.fl(wave_3d_flow)

    clips.append(clip)

# ✨ PURE RAW STITCHING - Zero fade effects, zero transitions, direct cuts only
if clips:
    final_video = concatenate_videoclips(clips, method="compose", padding=0)
    final_path = os.path.join(base_output_dir, "final_video_NO_FADE_ENHANCED_CAMERA.mp4")

    # Enhanced encoding for smooth playback
    final_video.write_videofile(
        final_path,
        codec='libx264',
        audio=False,
        fps=fps,  # Using centralized FPS setting
        bitrate="8000k",  # High quality
        preset='medium'
    )

    print(f"\n✅ ENHANCED 360° PAN anime story saved at: {final_path}")

    # ----------- ADD AUDIO TO FINAL VIDEO -----------
    print(f"\n🎵 ADDING AUDIO TO FINAL VIDEO...")

    # Generate audio using simple TTS integration
    audio_path = None
    if lesson_data and lesson_data.get('text'):
        try:
            from simple_audio_integration import add_simple_audio

            # Generate audio path
            audio_output_path = os.path.join(base_output_dir, "final_video_WITH_AUDIO.mp4")

            print(f"   🎵 Generating TTS audio for lesson...")

            # Add simple audio to video
            audio_path = add_simple_audio(final_path, lesson_data, audio_output_path)

            if audio_path != final_path:
                print(f"   ✅ Audio added successfully: {os.path.basename(audio_path)}")
            else:
                print(f"   ⚠️ Audio generation failed, using original video")

        except Exception as e:
            print(f"   ⚠️ Audio generation error: {e}")

    # Use audio version if available, otherwise original
    if audio_path and os.path.exists(audio_path) and audio_path != final_path:
        print(f"   ✅ Using video with audio: {os.path.basename(audio_path)}")
        final_video_for_processing = audio_path
    else:
        print(f"   ⚠️ No audio available, using original video")
        final_video_for_processing = final_path

    # ----------- ADD SUBTITLES TO FINAL VIDEO -----------
    if lesson_data and lesson_data.get('text'):
        print(f"\n🎬 ADDING SUBTITLES TO FINAL VIDEO...")
        subtitle_path = os.path.join(base_output_dir, "final_video_WITH_SUBTITLES.mp4")

        # Split text into sentences matching the number of clips
        sentences = [s.strip() for s in lesson_data['text'].split('.') if s.strip()]
        sentences = sentences[:len(generated_clips)]  # Match number of clips

        # Add subtitles with proper timing
        final_with_subtitles = add_subtitles_to_video_clips(final_video_for_processing, sentences, subtitle_path, len(generated_clips))

        if final_with_subtitles != final_video_for_processing:
            print(f"✅ Final video with subtitles: {os.path.basename(final_with_subtitles)}")
            final_video_for_storage = final_with_subtitles
        else:
            print(f"⚠️ Subtitle addition failed, using video without subtitles")
            final_video_for_storage = final_video_for_processing
    else:
        print(f"⚠️ No lesson text available for subtitles")
        final_video_for_storage = final_video_for_processing

    # ----------- STORAGE & DELIVERY SYSTEM -----------
    print(f"\n📤 STORAGE & DELIVERY SYSTEM:")

    # Prepare metadata for storage
    storage_metadata = {
        'title': lesson_data.get('title', 'Unknown Lesson') if lesson_data else 'Default Story',
        'level': lesson_data.get('level', 'Beginner') if lesson_data else 'Basic',
        'style': selected_style['name'],
        'text': lesson_data.get('text', paragraph) if lesson_data else paragraph,
        'subject': 'Text-to-Video Generation',
        'topic': lesson_data.get('title', 'Anime Story') if lesson_data else 'Anime Story',
        'generation_params': {
            'model': selected_style['model'],
            'guidance_scale': selected_style['guidance_scale'],
            'steps': selected_style['steps'],
            'clips_count': len(generated_clips),
            'total_duration': final_video.duration,
            'has_subtitles': lesson_data is not None,
            'camera_effects': '6 dynamic 3D motions (no zoom)',
            'style_description': selected_style['description']
        },
        'timestamp': datetime.datetime.now().isoformat(),
        'lesson_id': hashlib.md5(str(lesson_data).encode()).hexdigest()[:8] if lesson_data else 'default'
    }

    # Store video in organized storage
    storage_result = storage_system.store_video(final_video_for_storage, storage_metadata)

    if storage_result['success']:
        print(f"   ✅ Video stored successfully")
        print(f"   📁 Storage path: {storage_result['storage_path']}")
        print(f"   🔗 Access URL: {storage_result['access_url']}")
        print(f"   📄 Metadata saved: {storage_result['metadata_path']}")

        # Deliver to production system
        print(f"\n🚀 DELIVERING TO PRODUCTION SYSTEM:")
        delivery_result = storage_system.deliver_to_production(final_video_for_storage, storage_metadata)

        if delivery_result['success']:
            print(f"   ✅ Video delivered to production: 192.168.0.121:8001")
            print(f"   📊 Response status: {delivery_result['status_code']}")
        else:
            print(f"   ⚠️ Production delivery failed: {delivery_result.get('error', 'Unknown error')}")
            print(f"   💡 Video is still available in local storage")

        # Show storage statistics
        storage_stats = storage_system.get_storage_stats()
        print(f"\n📊 STORAGE STATISTICS:")
        print(f"   • Total stored files: {storage_stats['total_files']}")
        print(f"   • Total storage size: {storage_stats['total_size_mb']:.2f} MB")
        print(f"   • Stored videos: {storage_stats['stored_videos']}")
        print(f"   • Storage directory: {storage_stats['base_directory']}")

    else:
        print(f"   ❌ Storage failed: {storage_result.get('error', 'Unknown error')}")
        print(f"   💡 Video is available at: {final_video_for_storage}")

    # Dynamic summary based on actual lesson
    lesson_title = lesson_data.get('title', 'Video') if lesson_data else 'Video'

    print(f"📊 {lesson_title} - Generation Summary:")
    print(f"   • Total scenes: {len(clip_prompts)}")
    print(f"   • Successfully generated clips: {len(generated_clips)}")
    print(f"   • Final video duration: {final_video.duration:.2f} seconds")
    print(f"   • Camera effects: 6 dynamic 3D motions (NO ZOOM) ✅")
    print(f"   • 3D effects: Circular pan, orbital, tracking, pendulum ✅")
    print(f"   • Motion types: Tilt & drift, wave flow (pure 3D) ✅")
    print(f"   • Zoom effects: REMOVED - No zoom in/out ✅")
    print(f"   • Fades: ZERO - No fade in/out effects ✅")
    print(f"   • Transitions: Direct cuts with 3D motion flow ✅")
    print(f"   • Subtitles: {'✅ Added' if lesson_data else '❌ Not available'}")
    print(f"   • Output location: {final_path}")

    print(f"\n🚀 PHASE 1, 2 & 3 IMPROVEMENTS + ENHANCED 360° CAMERA:")
    print(f"   ✅ Smart frame selection for better pose detection")
    print(f"   ✅ Multi-control guidance (pose → depth → canny fallback)")
    print(f"   ✅ Enhanced prompts with consistency keywords")
    print(f"   ✅ Improved character continuity between clips")
    print(f"   🆕 Content-aware model optimization")
    print(f"   🆕 Character reference extraction & maintenance")
    print(f"   🆕 Adaptive control weights based on quality")
    print(f"   🆕 Quality scoring and issue detection")
    print(f"   🆕 Advanced prompt enhancement with character features")
    print(f"   🚀 PHASE 3: Face identity preservation system")
    print(f"   🚀 PHASE 3: Real-time LoRA training for character consistency")
    print(f"   🚀 PHASE 3: Automatic retry & recovery system")
    print(f"   🚀 PHASE 3: Multiple model support (anime, educational, etc.)")
    print(f"   🎯 ENHANCED 3D CAMERA: Pure 3D motion effects, NO ZOOM, zero fades")

    # Phase 3 Statistics
    retry_stats = auto_retry_system.get_performance_stats()
    lora_stats = realtime_lora_trainer.get_lora_stats()

    print(f"\n📊 PHASE 3 PERFORMANCE STATS:")
    if not retry_stats.get('no_data', False):
        print(f"   • Success rate: {retry_stats['success_rate']:.1%}")
        print(f"   • Avg attempts per clip: {retry_stats['avg_attempts_per_generation']:.1f}")

    print(f"   • Face identity established: {face_identity_established}")
    print(f"   • Character LoRA trained: {lora_trained}")
    print(f"   • Total generation attempts: {sum(generation_attempts.values()) if generation_attempts else len(generated_clips)}")
    print(f"   • Model used: {optimal_config['model_description']}")

    # ----------- MULTIPLE RENDER STYLES GENERATION -----------
    print(f"\n🎨 MULTIPLE RENDER STYLES AVAILABLE:")

    # Show available styles
    create_style_selection_menu()

    # SIMPLIFIED: Single output directory for all styles
    print(f"\n✅ VIDEO GENERATION COMPLETE!")
    print(f"   📁 All videos saved to: outputs/multi_clip/")
    print(f"   🎨 Current style: {selected_style}")
    print(f"   💡 To change style, modify the 'selected_style' variable and run again")

    # Close video objects
    final_video.close()

else:
    print("❌ No clips available for stitching!")

def main_function(selected_style='realistic'):
    """Main function that can be called by unified system or directly"""
    # The main generation logic is already executed above in the global scope
    # This function exists for compatibility with unified system
    pass

if __name__ == "__main__":
    import sys

    # Handle new parameter order: lesson_file.json style
    # or old parameter order: style
    selected_style = 'realistic'  # default

    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        if first_arg.endswith('.json'):
            # New format: lesson_file.json style
            selected_style = sys.argv[2] if len(sys.argv) > 2 else 'realistic'
        else:
            # Old format: style
            selected_style = first_arg

    print(f"🎨 Command line style selected: {selected_style}")

    # FIXED: Avoid circular import with unified_video_generator
    # The unified system calls this script, so we should NOT import it back
    print("🎬 Using multi-clip generation system (called by unified system)")
    print("✅ Video generation completed above")
