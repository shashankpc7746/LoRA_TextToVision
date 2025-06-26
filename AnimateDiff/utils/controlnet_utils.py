# utils/controlnet_utils.py

from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np
import torch

# Load OpenPose model once
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

def generate_openpose_image(input_image_path: str, output_path: str = "pose_output.png") -> str:
    """
    Given an image path, generate an OpenPose image using controlnet_aux.
    Saves the output pose image and returns its path.
    """
    image = Image.open(input_image_path).convert("RGB")
    pose_image = pose_detector(image)
    
    # Convert to uint8 image and save
    pose_np = np.array(pose_image)
    pose_image_pil = Image.fromarray(pose_np)
    pose_image_pil.save(output_path)
    
    return output_path
