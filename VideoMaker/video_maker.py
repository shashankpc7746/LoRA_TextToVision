# video_maker.py

import os
from moviepy.editor import ImageClip, concatenate_videoclips, vfx

def create_video_from_images(image_folder, output_path, fps=1):
    """
    Converts a sequence of images in a folder to an MP4 video with slight zoom effect.

    Parameters:
    - image_folder (str): Path to the folder containing image frames.
    - output_path (str): Path to save the output video file.
    - fps (int): Frames per second (use 1 or 2 for story-like pace).
    """
    print(f"📂 Reading images from: {image_folder}")

    # Get list of image files in sorted order
    images = sorted([
        os.path.join(image_folder, img)
        for img in os.listdir(image_folder)
        if img.endswith(".png")
    ])

    if not images:
        raise ValueError("❌ No .png images found in the folder.")

    print(f"🖼️ Total {len(images)} frames found.")
    print("🎞️ Creating animated video...")

    clips = []
    for img_path in images:
        clip = (
            ImageClip(img_path)
            .resize(width=720)  # Resize image for consistent size
            .set_duration(3)  # Each image stays for 3 seconds
            .fx(vfx.resize, lambda t: 1 + 0.01 * t)  # gentler Zoom in slightly over time
            .fadein(0.5)  # Optional: fade-in effect
        )
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, fps=24, codec="libx264")

    print(f"✅ Video saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    scene_number = 1  # Change based on the scene you generated
    input_folder = f"./frames_scene_{scene_number}"
    output_file = f"video_outputs/anime_scene_{scene_number}.mp4"

    os.makedirs("video_outputs", exist_ok=True)
    create_video_from_images(input_folder, output_file, fps=1)
