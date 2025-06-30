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
paragraph = """A wise old wizard walks slowly through a foggy enchanted forest. The wizard raises his staff, summoning glowing fireflies around him. The wizard opens an ancient scroll glowing with runes. As thunder roars, the wizard casts a protection spell into the sky. The wizard walks toward a glowing portal that appears before him. Calmly, the wizard steps into the portal and vanishes in a burst of light."""

base_output_dir = "outputs/multi_clip/"
os.makedirs(base_output_dir, exist_ok=True)
base_seed = 566565
clip_prompts = []

# ------------- STEP 1: Split Paragraph into Sub-Prompts -------------
def split_paragraph(text):
    """Split paragraph by sentence punctuation."""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

clip_prompts = split_paragraph(paragraph)
print(f"🧠 Detected {len(clip_prompts)} prompts:")
for i, p in enumerate(clip_prompts):
    print(f"   [{i+1}] {p}")

# ------------- STEP 2: Generate Clips One by One -------------
last_frame_path = None
for idx, prompt in enumerate(clip_prompts):
    print(f"\n🎬 Generating Clip {idx + 1}...")

    # Output paths
    clip_name = f"clip{idx + 1}"
    output_video = os.path.join(base_output_dir, f"{clip_name}.mp4")
    last_frame_path_new = os.path.join(base_output_dir, f"{clip_name}_last.png")
    pose_path = None

    # If not first clip, generate OpenPose from previous clip's last frame
    if last_frame_path:
        pose_path = generate_openpose_image(
            last_frame_path,
            os.path.join(base_output_dir, f"{clip_name}_pose.png")
        )

    # Run AnimateDiff
    generate_clip(
        prompt=prompt,
        output_path=output_video,
        pose_path=pose_path,
        init_image_path=last_frame_path,
        seed=base_seed + idx
    )

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
print("\n🎞️ Stitching all clips into one final video...")

clips = []
for idx in range(len(clip_prompts)):
    clip_path = os.path.join(base_output_dir, f"clip{idx + 1}.mp4")
    clip = VideoFileClip(clip_path)

    # Add fade
    if idx > 0:
        clip = clip.fx(vfx.fadein, 0.3)
    if idx < len(clip_prompts) - 1:
        clip = clip.fx(vfx.fadeout, 0.3)

    # 🔍 Add dynamic zoom (Ken Burns effect)
    if idx % 2 == 0:
        clip = clip.fx(vfx.resize, lambda t: 1 + 0.01 * t)  # Zoom in
        clip = crop(clip, x1=10, x2=clip.w)  # Pan from left
    else:
        clip = clip.fx(vfx.resize, lambda t: 1.1 - 0.01 * t)  # Zoom out
        clip = crop(clip, x1=0, x2=clip.w - 10)  # Pan to left

    clips.append(clip)

final_video = concatenate_videoclips(clips, method="compose")
final_path = os.path.join(base_output_dir, "final_output_stitched.mp4")
final_video.write_videofile(final_path, codec='libx264', audio=False)

print(f"\n✅ Final stitched video saved at: {final_path}")