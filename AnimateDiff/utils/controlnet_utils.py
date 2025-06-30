# utils/controlnet_utils.py

from controlnet_aux import OpenposeDetector, MidasDetector
from PIL import Image
import numpy as np
import torch
import os

# Load OpenPose and Depth models once
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
depth_estimator = MidasDetector.from_pretrained("lllyasviel/ControlNet")

def generate_openpose_image(input_image_path: str, output_path: str = "pose_output.png") -> str:
    """
    Generate OpenPose image from input. Returns saved path.
    """
    image = Image.open(input_image_path).convert("RGB")
    pose_image = pose_detector(image)

    pose_np = np.array(pose_image)
    Image.fromarray(pose_np).save(output_path)
    return output_path

def generate_depth_map(input_image_path: str, output_path: str = "depth_output.png") -> str:
    """
    Generate depth map using controlnet_aux. Returns saved path.
    """
    image = Image.open(input_image_path).convert("RGB")
    depth_image = depth_estimator(image)

    depth_np = np.array(depth_image)
    depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)

    Image.fromarray(depth_np).save(output_path)
    return output_path

def is_blank_pose_image(pose_path: str, threshold: float = 5.0) -> bool:
    """
    Check if pose image is essentially blank (low intensity).
    Returns True if blank.
    """
    img = Image.open(pose_path).convert("L")
    avg_pixel = np.array(img).mean()
    return avg_pixel < threshold
