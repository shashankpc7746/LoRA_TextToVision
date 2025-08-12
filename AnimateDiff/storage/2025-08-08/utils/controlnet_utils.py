from controlnet_aux import OpenposeDetector, ZoeDetector, CannyDetector
from PIL import Image, ImageFilter
import numpy as np
import torch
import cv2
import os
from sklearn.cluster import KMeans
import pickle

# Load detectors once
pose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators")
canny_detector = CannyDetector()

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

def generate_multi_control_guidance(input_image_path: str, base_output_path: str) -> dict:
    """
    Generate multiple types of control guidance for better consistency.
    Returns dict with available control types and their paths.
    """
    if isinstance(input_image_path, str):
        image = Image.open(input_image_path).convert("RGB")
    else:
        # Assume it's already a PIL Image or numpy array
        if isinstance(input_image_path, np.ndarray):
            image = Image.fromarray(input_image_path)
        else:
            image = input_image_path

    controls = {}
    base_name = base_output_path.replace('.png', '')

    # 1. Try OpenPose first (best for human consistency)
    try:
        pose_image = pose_detector(image)
        pose_np = np.array(pose_image)
        pose_pil = Image.fromarray(pose_np)

        if not is_blank_image(pose_pil):
            pose_path = f"{base_name}_pose.png"
            pose_pil.save(pose_path)
            controls['pose'] = pose_path
            print("✅ OpenPose guidance generated")
        else:
            print("⚠️ OpenPose result is blank")
    except Exception as e:
        print(f"❌ OpenPose failed: {e}")

    # 2. Generate Depth (good fallback for spatial consistency)
    try:
        depth_image = depth_detector(image)
        depth_path = f"{base_name}_depth.png"
        depth_image.save(depth_path)
        controls['depth'] = depth_path
        print("✅ Depth guidance generated")
    except Exception as e:
        print(f"❌ Depth generation failed: {e}")

    # 3. Generate Canny edges (good for structural consistency)
    try:
        canny_image = canny_detector(image)
        canny_path = f"{base_name}_canny.png"
        canny_image.save(canny_path)
        controls['canny'] = canny_path
        print("✅ Canny guidance generated")
    except Exception as e:
        print(f"❌ Canny generation failed: {e}")

    # Return the best available control
    if 'pose' in controls:
        primary_control = controls['pose']
        control_type = 'pose'
    elif 'depth' in controls:
        primary_control = controls['depth']
        control_type = 'depth'
    elif 'canny' in controls:
        primary_control = controls['canny']
        control_type = 'canny'
    else:
        primary_control = None
        control_type = None

    print(f"🎯 Using {control_type} as primary control")

    return {
        'primary_control': primary_control,
        'control_type': control_type,
        'all_controls': controls
    }

# ------------- PHASE 2: CHARACTER REFERENCE EXTRACTION -------------

def extract_character_features(image_path, output_dir):
    """Extract and save character features for consistency"""

    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path

    features = {}

    # 1. Extract dominant colors (clothing, skin tone)
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)

    # Use KMeans to find dominant colors
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    features['dominant_colors'] = dominant_colors.tolist()

    # 2. Extract face region (if human)
    face_region = extract_face_region(image)
    if face_region is not None:
        features['has_face'] = True
        features['face_colors'] = extract_face_colors(face_region)
    else:
        features['has_face'] = False

    # 3. Extract clothing/texture patterns
    clothing_features = extract_clothing_features(image)
    features.update(clothing_features)

    # 4. Extract pose/body structure
    pose_features = extract_pose_features(image)
    features.update(pose_features)

    # Save features
    features_path = os.path.join(output_dir, "character_features.pkl")
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    print(f"✅ Character features extracted and saved to {features_path}")
    return features

def extract_face_region(image):
    """Extract face region using simple color-based detection"""
    img_array = np.array(image)
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

    # Define skin tone range
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find largest contour (likely face)
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract face region with some padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_array.shape[1], x + w + padding)
        y2 = min(img_array.shape[0], y + h + padding)

        face_region = img_array[y1:y2, x1:x2]
        return Image.fromarray(face_region)

    return None

def extract_face_colors(face_image):
    """Extract dominant colors from face region"""
    face_array = np.array(face_image)
    pixels = face_array.reshape(-1, 3)

    # Use KMeans for face colors
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    face_colors = kmeans.cluster_centers_.astype(int)

    return face_colors.tolist()

def extract_clothing_features(image):
    """Extract clothing and texture features"""
    img_array = np.array(image)

    # Focus on middle region (likely clothing)
    h, w = img_array.shape[:2]
    clothing_region = img_array[h//4:3*h//4, w//4:3*w//4]

    # Extract texture features using edge detection
    gray = cv2.cvtColor(clothing_region, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Extract clothing colors
    clothing_pixels = clothing_region.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(clothing_pixels)
    clothing_colors = kmeans.cluster_centers_.astype(int)

    return {
        'clothing_colors': clothing_colors.tolist(),
        'texture_density': float(edge_density)
    }

def extract_pose_features(image):
    """Extract basic pose/body structure features"""
    try:
        pose_image = pose_detector(image)
        pose_array = np.array(pose_image)

        # Calculate pose density and distribution
        pose_pixels = np.sum(pose_array > 10)
        total_pixels = pose_array.shape[0] * pose_array.shape[1]
        pose_density = pose_pixels / total_pixels

        # Find pose center of mass
        y_coords, x_coords = np.where(pose_array > 10)
        if len(x_coords) > 0:
            center_x = np.mean(x_coords) / pose_array.shape[1]
            center_y = np.mean(y_coords) / pose_array.shape[0]
        else:
            center_x, center_y = 0.5, 0.5

        return {
            'pose_density': float(pose_density),
            'pose_center': [float(center_x), float(center_y)]
        }
    except:
        return {
            'pose_density': 0.0,
            'pose_center': [0.5, 0.5]
        }

def load_character_features(output_dir):
    """Load saved character features"""
    features_path = os.path.join(output_dir, "character_features.pkl")

    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        print(f"✅ Character features loaded from {features_path}")
        return features
    else:
        print(f"⚠️ No character features found at {features_path}")
        return None

def enhance_prompt_with_character_features(prompt, character_features):
    """Enhance prompt with specific character features"""
    if character_features is None:
        return prompt

    enhancements = []

    # Add color consistency
    if 'dominant_colors' in character_features:
        colors = character_features['dominant_colors']
        # Convert RGB to color names (simplified)
        color_descriptions = []
        for color in colors[:2]:  # Use top 2 colors
            r, g, b = color
            if r > 150 and g > 150 and b > 150:
                color_descriptions.append("light colored")
            elif r < 100 and g < 100 and b < 100:
                color_descriptions.append("dark colored")
            elif r > g and r > b:
                color_descriptions.append("reddish")
            elif g > r and g > b:
                color_descriptions.append("greenish")
            elif b > r and b > g:
                color_descriptions.append("bluish")

        if color_descriptions:
            enhancements.append(f"wearing {', '.join(color_descriptions)} clothing")

    # Add face consistency
    if character_features.get('has_face', False):
        enhancements.extend([
            "same facial features",
            "consistent face",
            "identical person"
        ])

    # Add texture consistency
    if 'texture_density' in character_features:
        if character_features['texture_density'] > 0.1:
            enhancements.append("detailed textured clothing")
        else:
            enhancements.append("smooth clothing")

    # Combine with original prompt
    if enhancements:
        enhanced_prompt = f"{prompt}, {', '.join(enhancements)}"
        return enhanced_prompt

    return prompt

def calculate_character_consistency_score(current_frame, character_features):
    """Calculate how consistent current frame is with character features"""
    if character_features is None:
        return 0.5  # Neutral score

    try:
        current_features = extract_character_features(current_frame, "/tmp")

        score = 0.0
        weight_sum = 0.0

        # Compare dominant colors
        if 'dominant_colors' in current_features and 'dominant_colors' in character_features:
            color_similarity = calculate_color_similarity(
                current_features['dominant_colors'],
                character_features['dominant_colors']
            )
            score += color_similarity * 0.4
            weight_sum += 0.4

        # Compare face features
        if character_features.get('has_face', False) and current_features.get('has_face', False):
            face_similarity = calculate_color_similarity(
                current_features['face_colors'],
                character_features['face_colors']
            )
            score += face_similarity * 0.3
            weight_sum += 0.3

        # Compare clothing features
        if 'clothing_colors' in current_features and 'clothing_colors' in character_features:
            clothing_similarity = calculate_color_similarity(
                current_features['clothing_colors'],
                character_features['clothing_colors']
            )
            score += clothing_similarity * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.5

    except Exception as e:
        print(f"⚠️ Error calculating consistency score: {e}")
        return 0.5

def calculate_color_similarity(colors1, colors2):
    """Calculate similarity between two sets of colors"""
    if not colors1 or not colors2:
        return 0.0

    # Convert to numpy arrays
    c1 = np.array(colors1)
    c2 = np.array(colors2)

    # Calculate minimum distances between color sets
    similarities = []
    for color1 in c1:
        min_dist = float('inf')
        for color2 in c2:
            # Euclidean distance in RGB space
            dist = np.sqrt(np.sum((color1 - color2) ** 2))
            min_dist = min(min_dist, dist)

        # Convert distance to similarity (0-1)
        similarity = max(0, 1 - (min_dist / 441.67))  # 441.67 is max RGB distance
        similarities.append(similarity)

    return np.mean(similarities)

# ------------- PHASE 2: ADAPTIVE CONTROL WEIGHTS -------------

def calculate_adaptive_control_weight(control_image, content_analysis, base_weight=0.8):
    """Calculate adaptive control weight based on control quality and content"""

    if control_image is None:
        return base_weight * 0.5  # Reduce weight if no control

    # Load control image
    if isinstance(control_image, str):
        control_img = Image.open(control_image).convert("RGB")
    else:
        control_img = control_image

    control_array = np.array(control_img)

    # Calculate control quality metrics
    quality_score = calculate_control_quality(control_array)

    # Adjust based on content type
    content_modifier = get_content_control_modifier(content_analysis)

    # Calculate final weight
    adaptive_weight = base_weight * quality_score * content_modifier

    # Clamp to reasonable range
    adaptive_weight = max(0.3, min(1.0, adaptive_weight))

    print(f"🎛️ Adaptive control weight: {adaptive_weight:.3f} (quality: {quality_score:.3f}, content: {content_modifier:.3f})")

    return adaptive_weight

def calculate_control_quality(control_array):
    """Calculate quality score of control image"""

    # Convert to grayscale for analysis
    if len(control_array.shape) == 3:
        gray = cv2.cvtColor(control_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = control_array

    # Calculate various quality metrics

    # 1. Information content (non-zero pixels)
    non_zero_ratio = np.sum(gray > 10) / (gray.shape[0] * gray.shape[1])

    # 2. Edge density (structural information)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # 3. Contrast (dynamic range)
    contrast = np.std(gray) / 255.0

    # 4. Spatial distribution (not too concentrated)
    # Calculate center of mass
    y_coords, x_coords = np.where(gray > 10)
    if len(x_coords) > 0:
        center_x = np.mean(x_coords) / gray.shape[1]
        center_y = np.mean(y_coords) / gray.shape[0]

        # Penalize if too concentrated in corners
        center_penalty = 1.0
        if center_x < 0.2 or center_x > 0.8 or center_y < 0.2 or center_y > 0.8:
            center_penalty = 0.8
    else:
        center_penalty = 0.5

    # Combine metrics
    quality_score = (
        non_zero_ratio * 0.3 +
        edge_density * 0.3 +
        contrast * 0.2 +
        center_penalty * 0.2
    )

    return min(1.0, quality_score)

def get_content_control_modifier(content_analysis):
    """Get control weight modifier based on content type"""

    if content_analysis is None:
        return 1.0

    primary_type = content_analysis.get('primary_type', 'object')
    motion_intensity = content_analysis.get('motion_intensity', 'medium')
    complexity = content_analysis.get('complexity', 'medium')

    modifier = 1.0

    # Adjust based on content type
    if primary_type == 'human':
        modifier *= 1.2  # Humans need stronger control
    elif primary_type == 'animal':
        modifier *= 0.9  # Animals need more freedom
    elif primary_type == 'object':
        modifier *= 1.1  # Objects benefit from structure

    # Adjust based on motion intensity
    if motion_intensity == 'high':
        modifier *= 0.8  # High motion needs less rigid control
    elif motion_intensity == 'low':
        modifier *= 1.1  # Low motion can handle more control

    # Adjust based on complexity
    if complexity == 'high':
        modifier *= 1.1  # Complex scenes need more guidance
    elif complexity == 'low':
        modifier *= 0.9  # Simple scenes need less control

    return modifier

def generate_adaptive_multi_control_guidance(input_image_path, base_output_path, content_analysis=None):
    """Enhanced multi-control guidance with adaptive weights"""

    # Generate base controls
    control_result = generate_multi_control_guidance(input_image_path, base_output_path)

    if control_result['primary_control'] is None:
        return control_result

    # Calculate adaptive weight
    adaptive_weight = calculate_adaptive_control_weight(
        control_result['primary_control'],
        content_analysis
    )

    # Add adaptive weight to result
    control_result['adaptive_weight'] = adaptive_weight
    control_result['content_analysis'] = content_analysis

    return control_result
