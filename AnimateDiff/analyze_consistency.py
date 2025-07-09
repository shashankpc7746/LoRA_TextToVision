#!/usr/bin/env python3
"""
Analyze consistency issues in generated multi-clip videos
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_image_properties(image_path):
    """Analyze basic properties of an image"""
    if not os.path.exists(image_path):
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB for analysis
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return {
        'path': image_path,
        'shape': img.shape,
        'mean_brightness': np.mean(img_rgb),
        'std_brightness': np.std(img_rgb),
        'dominant_colors': get_dominant_colors(img_rgb),
        'has_human_like_features': detect_human_features(img_rgb)
    }

def get_dominant_colors(img_rgb, k=3):
    """Get dominant colors in the image"""
    data = img_rgb.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    return centers.astype(int)

def detect_human_features(img_rgb):
    """Simple heuristic to detect if image likely contains human features"""
    # Convert to HSV for skin tone detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define skin tone range (rough approximation)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_percentage = np.sum(skin_mask > 0) / (img_rgb.shape[0] * img_rgb.shape[1])
    
    return skin_percentage > 0.02  # If more than 2% skin-like pixels

def analyze_pose_skeleton(pose_path):
    """Analyze pose skeleton image"""
    if not os.path.exists(pose_path):
        return None
    
    img = cv2.imread(pose_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Count non-black pixels (skeleton pixels)
    skeleton_pixels = np.sum(img > 10)  # Threshold for non-black
    total_pixels = img.shape[0] * img.shape[1]
    skeleton_density = skeleton_pixels / total_pixels
    
    # Find connected components (body parts)
    _, binary = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)
    
    return {
        'path': pose_path,
        'skeleton_density': skeleton_density,
        'num_body_parts': num_labels - 1,  # Subtract background
        'is_blank': skeleton_density < 0.001
    }

def main():
    base_dir = "outputs/multi_clip"
    
    print("🔍 ANALYZING MULTI-CLIP CONSISTENCY ISSUES")
    print("=" * 60)
    
    # Analyze the prompt structure
    paragraph = """
A photorealistic young woman in a dark blue hoodie walks slowly down a quiet, rain-soaked city street at dusk.
She looks tired but peaceful. Rain gently falls as her hoodie shifts in the soft wind.
She stops at a neon-lit vending machine, buys a warm canned drink, and warms her hands around it.
A small dog suddenly runs past, splashing water from a puddle. 
She smiles gently, as if reminded of something.
She continues walking and pauses at a glowing bakery window. 
The glass is fogged. Inside, a child waves at her.
She waves back softly, a quiet warmth in her eyes.
She walks on, approaching a train crossing as red warning lights begin to blink.
As the train speeds by behind her, she takes a sip of the warm drink, standing still in the gentle rain.
"""
    
    prompts = [line.strip() for line in paragraph.strip().split('\n') if line.strip()]
    print(f"📝 PROMPT ANALYSIS:")
    print(f"   • Total prompts: {len(prompts)}")
    print(f"   • Character consistency keywords: 'young woman', 'dark blue hoodie', 'photorealistic'")
    print(f"   • Scene changes: street → vending machine → bakery → train crossing")
    print()
    
    # Analyze last frames (character consistency)
    print("👤 CHARACTER CONSISTENCY ANALYSIS (Last Frames):")
    print("-" * 50)
    
    last_frame_analysis = []
    for i in range(1, 11):  # clip1 to clip10
        last_frame_path = os.path.join(base_dir, f"clip{i}_last.png")
        analysis = analyze_image_properties(last_frame_path)
        if analysis:
            last_frame_analysis.append(analysis)
            print(f"   Clip {i}: Brightness={analysis['mean_brightness']:.1f}, "
                  f"Std={analysis['std_brightness']:.1f}, "
                  f"Human-like={analysis['has_human_like_features']}")
    
    # Analyze pose skeletons
    print(f"\n🦴 POSE SKELETON ANALYSIS:")
    print("-" * 50)
    
    pose_analysis = []
    for i in range(2, 11):  # clip2 to clip10 (clip1 has no pose)
        pose_path = os.path.join(base_dir, f"clip{i}_pose.png")
        analysis = analyze_pose_skeleton(pose_path)
        if analysis:
            pose_analysis.append(analysis)
            print(f"   Clip {i} pose: Density={analysis['skeleton_density']:.4f}, "
                  f"Body parts={analysis['num_body_parts']}, "
                  f"Blank={analysis['is_blank']}")
    
    # Analyze depth images (fallback control)
    print(f"\n🌊 DEPTH IMAGE ANALYSIS:")
    print("-" * 50)
    
    depth_files = [f for f in os.listdir(base_dir) if f.endswith('_depth.png')]
    print(f"   • Found {len(depth_files)} depth images (fallback cases)")
    for depth_file in sorted(depth_files):
        clip_num = depth_file.split('_')[0]
        print(f"   • {clip_num}: OpenPose failed, used ZoeDepth instead")
    
    # Consistency issues analysis
    print(f"\n⚠️  IDENTIFIED CONSISTENCY ISSUES:")
    print("-" * 50)
    
    # Issue 1: No pose guidance for clip1
    print("   1. CLIP1 INITIALIZATION ISSUE:")
    print("      • Clip1 has no pose guidance (no previous frame)")
    print("      • Character appearance is purely from text prompt")
    print("      • This can cause initial character inconsistency")
    
    # Issue 2: Brightness variations
    if last_frame_analysis:
        brightness_values = [a['mean_brightness'] for a in last_frame_analysis]
        brightness_std = np.std(brightness_values)
        print(f"\n   2. LIGHTING CONSISTENCY:")
        print(f"      • Brightness variation across clips: {brightness_std:.2f}")
        if brightness_std > 20:
            print("      • HIGH variation detected - lighting inconsistency!")
        
    # Issue 3: Pose detection failures
    blank_poses = sum(1 for a in pose_analysis if a['is_blank'])
    print(f"\n   3. POSE DETECTION RELIABILITY:")
    print(f"      • {blank_poses}/{len(pose_analysis)} poses failed (fell back to depth)")
    if blank_poses > len(pose_analysis) * 0.3:
        print("      • HIGH failure rate - pose continuity compromised!")
    
    # Issue 4: Scene complexity changes
    print(f"\n   4. SCENE COMPLEXITY CHANGES:")
    print("      • Street scene (simple) → Vending machine (objects) → Bakery (complex)")
    print("      • Dog appearance in clip4 adds new element")
    print("      • Child in bakery window (clip7) changes scene dynamics")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    print("   1. Add initial pose guidance for clip1")
    print("   2. Improve OpenPose detection reliability")
    print("   3. Add character reference image consistency")
    print("   4. Implement face/identity preservation")
    print("   5. Use more consistent lighting prompts")
    print("   6. Consider LoRA training for character consistency")

if __name__ == "__main__":
    main()
