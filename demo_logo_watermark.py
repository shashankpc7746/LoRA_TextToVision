"""
Demo script to test logo-based visible watermarking
"""
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from security.visible_watermark import VisibleWatermarker, add_visible_watermark

print("\n" + "="*70)
print("LOGO WATERMARK DEMO")
print("="*70)

# Create test video
print("\n1️⃣  Creating test video...")
with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
    test_video = f.name

# Create a sample video with gradient background
width, height = 1280, 720
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(test_video, fourcc, fps, (width, height))

# Generate 90 frames (3 seconds at 30fps)
for i in range(90):
    # Create gradient frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Animated gradient
    offset = i * 3
    for y in range(height):
        for x in range(width):
            frame[y, x] = [
                (100 + (x + offset) // 10) % 256,  # Blue channel
                (150 + (y + offset) // 10) % 256,  # Green channel
                200  # Red channel
            ]
    
    # Add some text to make it look like content
    cv2.putText(
        frame,
        "Sample Educational Video Content",
        (width // 2 - 300, height // 2),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    out.write(frame)

out.release()
print(f"✅ Test video created: {os.path.basename(test_video)}")
print(f"   Resolution: {width}x{height}")
print(f"   Duration: 3 seconds @ {fps} fps")

# Test different opacity levels
print("\n" + "="*70)
print("2️⃣  Testing Logo Watermark with Different Opacity Levels")
print("="*70)

watermarker = VisibleWatermarker()

# Test 1: Subtle (15% opacity) - Production style
print("\n📹 Test 1: Subtle Logo (15% opacity - Production)")
print("-" * 70)
result_subtle = watermarker.add_corner_watermark(
    test_video,
    position="bottom-right",
    opacity=0.15,
    scale=0.08
)
print(f"✅ Output: {os.path.basename(result_subtle)}")

# Test 2: Moderate (30% opacity) - Free tier
print("\n📹 Test 2: Moderate Logo (30% opacity - Free Tier)")
print("-" * 70)
result_moderate = watermarker.add_corner_watermark(
    test_video,
    position="bottom-right",
    opacity=0.30,
    scale=0.10
)
print(f"✅ Output: {os.path.basename(result_moderate)}")

# Test 3: Prominent (50% opacity) - Demo mode
print("\n📹 Test 3: Prominent Logo (50% opacity - Demo)")
print("-" * 70)
result_prominent = watermarker.add_corner_watermark(
    test_video,
    position="top-right",
    opacity=0.50,
    scale=0.12
)
print(f"✅ Output: {os.path.basename(result_prominent)}")

# Test 4: Different positions
print("\n" + "="*70)
print("3️⃣  Testing Different Logo Positions")
print("="*70)

positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
for pos in positions:
    print(f"\n📹 Position: {pos}")
    result = watermarker.add_corner_watermark(
        test_video,
        output_path=test_video.replace('.mp4', f'_{pos}.mp4'),
        position=pos,
        opacity=0.25,
        scale=0.08
    )
    print(f"✅ Output: {os.path.basename(result)}")

# Summary
print("\n" + "="*70)
print("4️⃣  SUMMARY")
print("="*70)
print("""
✅ Logo watermarking successfully implemented!

Features:
- Uses company logo (BHI_logo.png) instead of text
- Adjustable opacity (0.0 to 1.0)
- Adjustable size (scale relative to video width)
- Multiple positions (4 corners)
- Maintains logo transparency (PNG alpha channel)
- Professional appearance

Recommended Settings:
- Production (Paid): 15% opacity, 8% scale, bottom-right
- Free Tier: 30% opacity, 10% scale, bottom-right
- Demo/Trial: 50% opacity, 12% scale, top-right

Test videos generated:
""")

# List generated files
if os.path.exists(result_subtle):
    print(f"  1. {os.path.basename(result_subtle)} (subtle)")
if os.path.exists(result_moderate):
    print(f"  2. {os.path.basename(result_moderate)} (moderate)")
if os.path.exists(result_prominent):
    print(f"  3. {os.path.basename(result_prominent)} (prominent)")

print("\n💡 You can review these videos to choose the best opacity level!")

# Cleanup option
print("\n" + "="*70)
print("CLEANUP")
print("="*70)
cleanup = input("\nDelete test videos? (y/n): ").lower()

if cleanup == 'y':
    # Delete all test files
    import glob
    test_files = glob.glob(test_video.replace('.mp4', '*'))
    for f in test_files:
        if os.path.exists(f):
            os.unlink(f)
            print(f"✅ Deleted: {os.path.basename(f)}")
    print("\n✅ Cleanup complete!")
else:
    print(f"\n📹 Test videos saved in: {os.path.dirname(test_video)}")
    print("   Review them to see different opacity levels!")

print("\n✅ Logo watermark demo complete!")
