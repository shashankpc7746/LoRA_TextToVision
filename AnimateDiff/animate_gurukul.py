import torch
import os
import sys
import json
from pathlib import Path

# Add the animatediff module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'animatediff'))

# Fix tqdm progress bar issues directly
def patch_tqdm():
    """Fix tqdm division by zero issue and enable white box progress bars"""
    try:
        import tqdm.std

        # Patch the specific method that causes division by zero
        original_format = tqdm.std.tqdm.format_meter

        @staticmethod
        def safe_format_meter(n, total, elapsed, ncols=None, prefix='', ascii=False, unit='it',
                            unit_scale=False, rate=None, bar_format=None, postfix=None,
                            unit_divisor=1000, initial=0, colour=None, **kwargs):

            # Always create a custom progress bar with white boxes
            if total and total > 0:
                percent = (n / total) * 100
                bar_length = 25
                filled_length = int(bar_length * n // total)

                # Use proper Unicode box characters
                filled_char = '█'  # Full block (white box)
                empty_char = '░'   # Light shade
                bar = filled_char * filled_length + empty_char * (bar_length - filled_length)

                # Calculate rate and ETA
                if elapsed > 0 and n > 0:
                    rate = n / elapsed
                    eta = (total - n) / rate if rate > 0 else 0
                    rate_str = f"{rate:.1f}{unit}/s"
                    eta_str = f"{eta:.0f}s"
                else:
                    rate_str = "?it/s"
                    eta_str = "?s"

                # Format like standard tqdm with visual bar
                return f"{prefix}: |{bar}| {percent:.0f}% {n}/{total} [{elapsed:.0f}s<{eta_str}, {rate_str}]"
            else:
                return f"{prefix}: {n} items"

        # Apply the patch safely
        try:
            tqdm.std.tqdm.format_meter = safe_format_meter
        except Exception as e:
            print(f"Warning: Could not patch tqdm format_meter: {e}")
            # Fallback to disabling tqdm
            import os
            os.environ['TQDM_DISABLE'] = '1'

    except Exception as e:
        # Fallback to disabling tqdm if patch fails
        import os
        os.environ['TQDM_DISABLE'] = '1'

# Apply the patch
patch_tqdm()

# Use enhanced diffusers AnimateDiff with better configuration
try:
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
    from diffusers.utils import export_to_video
    ANIMATEDIFF_AVAILABLE = True
    print("🚀 Using Enhanced Diffusers AnimateDiff with Character Consistency")
except ImportError:
    # Fallback imports for newer diffusers versions
    try:
        from diffusers.pipelines.animatediff.pipeline_animatediff import AnimateDiffPipeline
        from diffusers.models.unets.unet_motion_model import MotionAdapter
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
        from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
        from diffusers.utils.export_utils import export_to_video
        ANIMATEDIFF_AVAILABLE = True
        print("🚀 Using Enhanced Diffusers AnimateDiff (fallback imports)")
    except ImportError:
        print("⚠️ Warning: Could not import diffusers components. Some features may not work.")
        AnimateDiffPipeline = None
        MotionAdapter = None
        EulerDiscreteScheduler = None
        AutoencoderKL = None
        export_to_video = None
        ANIMATEDIFF_AVAILABLE = False
from PIL import Image
import diffusers.utils.logging
import re
diffusers.utils.logging.enable_progress_bar()

# REMOVED OLD HARDCODED OPTIMIZATION FUNCTIONS
# Now using content-aware enhancement from multi_clip_generator.py

# ===== ENHANCED ANIMATEDIFF CONFIGURATION =====
device = "cuda"
dtype = torch.float16
step = 30  # Optimized for speed vs quality balance
num_frames = 24  # Optimized for speed vs quality balance
fps = 12  # ← SINGLE FPS SETTING FOR ALL VIDEO GENERATION - Optimized for speed

# Model configurations - Using ONLY proven AnimateDiff-compatible models
MODEL_CONFIGS = {
    "realistic": {
        "base_model": "SG161222/Realistic_Vision_V5.1_noVAE",  # Better realistic model for face quality
        "vae_model": "stabilityai/sd-vae-ft-mse",
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2"  # Use stable motion adapter
    },
    "anime": {
        "base_model": "xyn-ai/anything-v4.0",  # Pure anime model for true anime style
        "vae_model": "stabilityai/sd-vae-ft-mse",
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2"  # Use stable motion adapter
    },
    "artistic": {
        "base_model": "runwayml/stable-diffusion-v1-5",  # Better general artistic base
        "vae_model": "stabilityai/sd-vae-ft-mse",
        "motion_adapter": "guoyww/animatediff-motion-adapter-v1-5-2"  # Use stable motion adapter
    }
}

# Global pipeline variable
pipe = None
current_style = None

def initialize_animatediff_pipeline(style="realistic"):
    """Initialize proper AnimateDiff pipeline with full repository features"""
    global pipe, current_style

    if current_style == style and pipe is not None:
        return pipe

    print(f"🚀 Initializing AnimateDiff Pipeline for style: {style}")
    config = MODEL_CONFIGS[style]

    # Use enhanced diffusers AnimateDiff with character consistency features
    print("🚀 Loading Enhanced Diffusers AnimateDiff...")

    # Load Motion Adapter with better configuration
    adapter = MotionAdapter.from_pretrained(
        config["motion_adapter"],
        torch_dtype=dtype
    ).to(device)

    # Load AnimateDiff pipeline with fallback for compatibility
    print(f"📦 Loading base model: {config['base_model']}")
    try:
        pipe = AnimateDiffPipeline.from_pretrained(
            config["base_model"],
            motion_adapter=adapter,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        print(f"✅ Successfully loaded: {config['base_model']}")
    except Exception as e:
        print(f"⚠️ Failed to load {config['base_model']}: {e}")
        print(f"🔄 Falling back to runwayml/stable-diffusion-v1-5")
        pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)

    # Replace VAE with better one
    pipe.vae = AutoencoderKL.from_pretrained(
        config["vae_model"],
        torch_dtype=dtype
    ).to(device)

    # Configure scheduler for better quality
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear"
    )

    # Enable optimizations
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()

    current_style = style
    print(f"✅ AnimateDiff pipeline initialized for {style}")
    return pipe

# 🔁 Enhanced function for multi-clip generation with character consistency
def generate_clip(prompt: str,
                  output_path: str,
                  pose_path: str = None,
                  init_image_path: str = None,
                  seed: int = 123,
                  style: str = "realistic",
                  negative_prompt: str = None):
    """
    Generate a 32-frame animation with enhanced character consistency
    """

    # Initialize pipeline for the requested style
    pipeline = initialize_animatediff_pipeline(style)

    generator = torch.Generator(device).manual_seed(seed)
    control_image = None

    if pose_path and os.path.exists(pose_path):
        control_image = Image.open(pose_path).convert("RGB").resize((512, 512))

    init_image = None
    if init_image_path and os.path.exists(init_image_path):
        init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

    # USE ENHANCED PROMPTS FROM MULTI_CLIP_GENERATOR - NO HARDCODED CONTENT!
    # The enhanced prompts are now generated dynamically by the style-specific enhancement functions
    # This ensures content-aware backgrounds and characters based on the actual lesson content
    enhanced_prompt = prompt  # Will be enhanced by the style-specific functions in multi_clip_generator.py

    # Style-specific negative prompts - AGGRESSIVE BACKGROUND FORCING
    if style == "anime":
        # ANIME NEGATIVE PROMPTS - FORCED BACKGROUND GENERATION
        negative_prompt = "blank background, empty background, white background, void background, no background, missing background, plain background, bare background, solid color background, monochrome background, no scenery, no landscape, no environment, no setting, indoor, inside, room, wall, ceiling, studio background, plain backdrop, multiple faces, many faces, face collage, repeated faces, duplicate faces, face repetition, 10 faces, many characters, multiple people, crowd of faces, face grid, face mosaic, different character each clip, changing character, different person, character inconsistency, new character, character change, appearance change, boy, male, man, masculine, static pose, standing still, no movement, no action, inactive, motionless, frozen pose, still pose, not moving, stationary, idle, passive, cropped body, half body, cut off, partial body, missing legs, missing arms, close up, zoomed in, tight crop, head only, upper body only, torso only, incomplete figure, deformed face, distorted face, ugly face, bad face, malformed face, disfigured face, bad anatomy, deformed body, distorted body, malformed body, broken anatomy, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, blurry, low quality, bad quality, poor quality, worst quality, no background elements, invisible background, transparent background, backgroundless, void scenery, empty landscape, missing environment, no setting details, no character motion, static character, frozen character, unclear character, blurry character, low character quality, poor character visibility, no character details, missing character features, character distortion, character deformation, dark background, dim lighting, poor background visibility, unclear background, pixelated, large pixels, low resolution, blurry details, unclear details, dark scenes, poorly lit, dim scenes"
    elif style == "realistic":
        # REALISTIC NEGATIVE PROMPTS - CRITICAL FACIAL DISTORTION FIXES
        negative_prompt = "distorted eyes, asymmetrical eyes, malformed eyes, crossed eyes, uneven eyes, deformed nose, crooked nose, malformed nose, distorted mouth, asymmetrical mouth, malformed lips, crooked smile, facial distortion, facial asymmetry, unnatural facial features, bad facial anatomy, deformed facial structure, static pose, standing still, no movement, no action, inactive, motionless, frozen pose, still pose, not moving, stationary, idle, passive, stiff posture, rigid body, blank background, empty background, white background, void background, no background, missing background, plain background, bare background, indoor, inside, room, wall, ceiling, anime, cartoon, animated, drawn, illustration, painting, sketch, boy, male, man, masculine, different character, changing character, different person, character change, appearance change, different hair, different clothes, different outfit, cropped body, half body, cut off, partial body, missing legs, missing arms, close up, zoomed in, tight crop, head only, upper body only, torso only, incomplete figure, deformed face, distorted face, ugly face, bad face, malformed face, disfigured face, bad anatomy, deformed body, distorted body, malformed body, broken anatomy, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, blurry, low quality, bad quality, poor quality, worst quality, unrealistic, fake, artificial"
    else:  # artistic
        # ARTISTIC NEGATIVE PROMPTS - FORCED BACKGROUND GENERATION
        negative_prompt = "blank background, empty background, white background, void background, no background, missing background, plain background, bare background, solid color background, monochrome background, no scenery, no landscape, no environment, no setting, indoor, inside, room, wall, ceiling, studio background, plain backdrop, high contrast painting, over-saturated colors, extreme contrast, harsh lighting, painting-like background, abstract background, unclear background elements, unidentifiable background, inconsistent character, changing character, different character each clip, character inconsistency, different person, facial inconsistency, different facial features, face change, identity change, appearance change, new character, character change, boy, male, man, masculine, static pose, standing still, no movement, no action, inactive, motionless, frozen pose, still pose, not moving, stationary, idle, passive, different hair, different clothes, different outfit, cropped body, half body, cut off, partial body, missing legs, missing arms, close up, zoomed in, tight crop, head only, upper body only, torso only, incomplete figure, deformed face, distorted face, ugly face, bad face, malformed face, disfigured face, bad anatomy, deformed body, distorted body, malformed body, broken anatomy, extra limbs, missing limbs, bad hands, deformed hands, extra fingers, missing fingers, blurry, low quality, bad quality, poor quality, worst quality, no background elements, invisible background, transparent background, backgroundless, void scenery, empty landscape, missing environment, no setting details, abstract art, unclear art, confusing art, messy art, realistic appearance, photorealistic, realistic style, realistic rendering, realistic texture, realistic lighting, realistic shadows, realistic anatomy, realistic proportions, realistic skin, realistic hair, realistic clothing, realistic background, jarring transitions, abrupt frame changes, choppy motion, inconsistent frame flow, rough transitions, frame jumps, motion stuttering, frame stuttering, character distortion, facial distortion, body distortion, anatomical distortion, pixelated characters, sparse pixels, unclear characters, low character quality, poor character visibility, no character details, missing character features, choppy frame transitions, rough motion flow, inconsistent motion, poor motion quality, low motion smoothness"

    print(f"🎞️ Generating {style} clip: {prompt[:60]}...")
    print(f"🎯 Enhanced prompt: {enhanced_prompt[:80]}...")

    # Use enhanced prompt directly - NO MORE HARDCODED OPTIMIZATION
    main_prompt = enhanced_prompt
    print(f"✅ Using content-aware enhanced prompt from multi_clip_generator")

    # Enhanced AnimateDiff with style-specific parameters for better quality
    # Get style-specific parameters from config
    style_config = MODEL_CONFIGS.get(style, MODEL_CONFIGS["realistic"])

    # Style-specific guidance scale for comprehensive improvements
    if style == "realistic":
        guidance_scale = 16  # Maintained for good framing control
        inference_steps = 38  # Increased for better movement quality
    elif style == "anime":
        guidance_scale = 16  # Optimized for better character clarity and reduced blur
        inference_steps = 50  # Increased for better quality and reduced pixelation
    else:  # artistic
        guidance_scale = 14  # Further reduced for better artistic style and character clarity
        inference_steps = 48  # Increased for better character consistency and artistic quality

    output = pipeline(
        prompt=main_prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        width=512,  # Stable resolution for consistent quality
        height=512,  # Stable resolution for consistent quality
        generator=generator
    )

    # Extract video from output and save with proper FPS
    # In diffusers 0.33.1, AnimateDiff uses 'frames' instead of 'videos'
    video = output.frames[0]
    export_to_video(video, output_path, fps=fps)

    print(f"✅ {style.title()} clip saved: {output_path}")
    return output_path

def download_motion_modules():
    """Download essential AnimateDiff motion modules for better consistency"""
    import requests
    from pathlib import Path

    motion_module_dir = Path("models/Motion_Module")
    motion_module_dir.mkdir(parents=True, exist_ok=True)

    # Essential motion modules for character consistency
    motion_modules = {
        "mm_sd_v15_v2.ckpt": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
        "v3_sd15_mm.ckpt": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
    }

    for filename, url in motion_modules.items():
        filepath = motion_module_dir / filename
        if not filepath.exists():
            print(f"📦 Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ {filename} already exists")

def download_domain_adapters():
    """Download domain adapters for artifact reduction"""
    import requests
    from pathlib import Path

    adapter_dir = Path("models/DreamBooth_LoRA")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Domain adapters for better quality
    adapters = {
        "v3_sd15_adapter.ckpt": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt"
    }

    for filename, url in adapters.items():
        filepath = adapter_dir / filename
        if not filepath.exists():
            print(f"📦 Downloading {filename}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"✅ Downloaded {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ {filename} already exists")

# Initialize downloads on import
if __name__ == "__main__":
    print("🚀 Initializing Enhanced AnimateDiff System...")
    download_motion_modules()
    download_domain_adapters()
    print("✅ AnimateDiff system ready!")
