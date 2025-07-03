from controlnet_aux import OpenposeDetector, ZoeDetector
from PIL import Image
import numpy as np
import torch
import cv2

# Load OpenPose and ZoeDepth once
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators")

def is_blank_image(pil_image: Image.Image, threshold: float = 10.0) -> bool:
    """
    Checks whether an image is mostly blank (black or low variance).
    """
    gray = np.array(pil_image.convert("L"))
    std = np.std(gray)
    return std < threshold

def generate_openpose_image(input_image_path: str, output_path: str = "pose_output.png") -> str:
    """
    Generates OpenPose image. Falls back to depth if result is blank.
    """
    image = Image.open(input_image_path).convert("RGB")
    pose_image = pose_detector(image)

    # Convert and check if blank
    pose_np = np.array(pose_image)
    pose_pil = Image.fromarray(pose_np)

    if is_blank_image(pose_pil):
        print("⚠️ OpenPose result is blank. Falling back to ZoeDepth...")
        pose_image = depth_detector(image)
        pose_np = np.array(pose_image)
        pose_pil = Image.fromarray(pose_np)

    pose_pil.save(output_path)
    return output_path
