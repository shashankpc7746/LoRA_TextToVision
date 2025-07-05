# multi_clip_generator.py

import os
import shutil
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from moviepy.video.fx.all import crop
from animate_gurukul import generate_clip
from utils.controlnet_utils import generate_openpose_image

# ----------- CONFIGURATION -----------
# paragraph = """A boy walks through a desert as sand swirls around him. He shields his eyes from the blazing sun and then picks up a strange glowing artifact. Suddenly, the sky turns purple and lightning flashes in the distance. The boy starts to run, clutching the artifact tightly."""
# paragraph = """A young girl stands in an abandoned Victorian mansion at night, holding a flickering candle. The wallpaper peels, revealing faces that shift and whisper. As thunder crashes outside, a shadowy figure appears behind her in the mirror, but when she turns—no one's there. Slowly, the walls begin to bleed ink, and the chandelier swings violently above her."""
# paragraph = """A curious monkey swings energetically through the dense jungle canopy. The monkey spots a tall banana tree glowing under the sunlight. With agility, the monkey climbs the tree, its tail curling around the branch. The monkey grabs a ripe banana and peels it with excitement. Joyfully, the monkey eats the banana while perched on a leafy branch. The monkey smiles happily, surrounded by birds and colorful butterflies."""
# paragraph = """The girl walks down a quiet city street at sunrise. The girl looks up at the tall buildings glowing in the morning light. The girl pulls out her camera and takes a photo. The girl hears music nearby and follows the sound. The girl smiles as she finds a street artist playing guitar on the corner."""
# paragraph = """A wise old wizard walks slowly through a foggy enchanted forest. The wizard raises his staff, summoning glowing fireflies around him. The wizard opens an ancient scroll glowing with runes. As thunder roars, the wizard casts a protection spell into the sky. The wizard walks toward a glowing portal that appears before him. Calmly, the wizard steps into the portal and vanishes in a burst of light."""
# paragraph = """The person stands at the edge of the pool, arms raised. The person takes a deep breath and runs forward. The person jumps high and flips into a somersault. The person spins through the air above the water. The person splashes into the pool with a big, clean dive."""
paragraph = """
Anime boy wearing a hoodie walks on a quiet street under a grey sky.
Rain falls gently on anime boy as soft wind moves the hoodie.
Anime boy stops at a glowing vending machine beside the road.
Anime boy buys a warm canned coffee and holds the coffee with both hands.
A small dog runs past anime boy, splashing water in anime style.
Anime boy smiles and starts walking again through the calm street.
Anime boy passes an anime bakery with warm yellow lights in the window.
Anime boy pauses and looks inside the bakery as steam fogs the glass.
Anime boy stands near a train crossing while red lights start flashing.
Anime boy drinks the coffee slowly as the train moves fast through the rain.
"""

base_output_dir = "outputs/multi_clip/"
os.makedirs(base_output_dir, exist_ok=True)
base_seed = 545456451
clip_prompts = []

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

# ------------- STEP 2: Generate Clips One by One -------------
print(f"\n🎬 Starting generation of {len(clip_prompts)} clips...")
last_frame_path = None
generated_clips = []

for idx, prompt in enumerate(clip_prompts):
    print(f"\n🎬 Generating Clip {idx + 1}/{len(clip_prompts)}...")
    print(f"📝 Prompt: {prompt}")

    # Output paths
    clip_name = f"clip{idx + 1}"
    output_video = os.path.join(base_output_dir, f"{clip_name}.mp4")
    last_frame_path_new = os.path.join(base_output_dir, f"{clip_name}_last.png")
    pose_path = None

    # If not first clip, generate OpenPose from previous clip's last frame
    if last_frame_path:
        print(f"🎯 Using continuity from previous clip...")
        pose_path = generate_openpose_image(
            last_frame_path,
            os.path.join(base_output_dir, f"{clip_name}_pose.png")
        )

    try:
        # Run AnimateDiff
        generate_clip(
            prompt=prompt,
            output_path=output_video,
            pose_path=pose_path,
            init_image_path=last_frame_path,
            seed=base_seed + idx
        )
        generated_clips.append(output_video)
        print(f"✅ Clip {idx + 1} generated successfully!")

    except Exception as e:
        print(f"❌ Error generating clip {idx + 1}: {str(e)}")
        print("⚠️ Continuing with next clip...")
        continue

    # ------------- STEP 3: Extract Last Frame of Clip -------------
    cap = cv2.VideoCapture(output_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite(last_frame_path_new, frame)
        last_frame_path = last_frame_path_new
    else:
        print(f"⚠️ Failed to extract last frame from {output_video}")
        break

# ------------- STEP 4: Stitch All Clips into Final Video -------------
print(f"\n🎞️ Stitching {len(generated_clips)} clips into one final video...")

if not generated_clips:
    print("❌ No clips were generated successfully. Exiting...")
    exit(1)

clips = []
for idx, clip_path in enumerate(generated_clips):
    if not os.path.exists(clip_path):
        print(f"⚠️ Skipping missing clip: {clip_path}")
        continue

    print(f"📎 Loading clip {idx + 1}: {os.path.basename(clip_path)}")
    clip = VideoFileClip(clip_path)

    # ⏳ Smooth fade in/out
    if idx > 0:
        clip = clip.fx(vfx.fadein, 0.6)
    if idx < len(generated_clips) - 1:
        clip = clip.fx(vfx.fadeout, 0.6)

    # 🔍 Enhanced dynamic zoom (Ken Burns) + pan
    if idx % 2 == 0:
        clip = clip.fx(vfx.resize, lambda t: 1.05 + 0.01 * t)  # Zoom in more
        clip = crop(clip, x1=20, x2=clip.w)  # Slight left pan
    else:
        clip = clip.fx(vfx.resize, lambda t: 1.15 - 0.01 * t)  # Zoom out more
        clip = crop(clip, x1=0, x2=clip.w - 20)  # Slight right pan

    clips.append(clip)

# ✨ Final stitching with NO black padding between clips
if clips:
    final_video = concatenate_videoclips(clips, method="compose", padding=0)
    final_path = os.path.join(base_output_dir, "final_output_stitched.mp4")
    final_video.write_videofile(final_path, codec='libx264', audio=False)

    print(f"\n✅ Final stitched video saved at: {final_path}")
    print(f"📊 Summary:")
    print(f"   • Total prompts: {len(clip_prompts)}")
    print(f"   • Successfully generated clips: {len(generated_clips)}")
    print(f"   • Final video duration: {final_video.duration:.2f} seconds")
    print(f"   • Output location: {final_path}")
else:
    print("❌ No clips available for stitching!")
