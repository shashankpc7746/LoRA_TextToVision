import torch
import os

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

        # Apply the patch
        tqdm.std.tqdm.format_meter = safe_format_meter

    except Exception as e:
        # Fallback to disabling tqdm if patch fails
        import os
        os.environ['TQDM_DISABLE'] = '1'

# Apply the patch
patch_tqdm()

from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import export_to_video
from PIL import Image

# Enable ASCII-only progress bars (no Unicode symbols)
import diffusers.utils.logging
diffusers.utils.logging.enable_progress_bar()

# ===== CENTRALIZED VIDEO SETTINGS =====
# Change FPS here and it will be used everywhere
device = "cuda"
dtype = torch.float16
step = 25
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model = "stabilityai/sd-vae-ft-mse"
num_frames = 32
fps = 8  # ← SINGLE FPS SETTING FOR ALL VIDEO GENERATION

# Load Motion Adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=dtype
).to(device)

# Load AnimateDiff pipeline with error handling
try:
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=dtype,
        local_files_only=False,
        use_safetensors=True
    ).to(device)
except Exception as e:
    print(f"Error loading pipeline: {e}")
    # Fallback without safetensors
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=dtype,
        local_files_only=False,
        use_safetensors=False
    ).to(device)

# Replace default VAE
pipe.vae = AutoencoderKL.from_pretrained(
    vae_model,
    torch_dtype=dtype
).to(device)

# Scheduler config
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config,
    timestep_spacing="trailing",
    beta_schedule="linear"
)

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# 🔁 Main function for multi-clip generation
def generate_clip(prompt: str,
                  output_path: str,
                  pose_path: str = None,
                  init_image_path: str = None,
                  seed: int = 123,
                  negative_prompt: str = "blurry, distorted, ghost eyes, unnatural skin, motion-blur, scene flicker , extra limbs, low quality"):
    """
    Generate a 32-frame animation with optional ControlNet and init_image.
    """

    generator = torch.Generator(device).manual_seed(seed)
    control_image = None

    if pose_path:
        control_image = Image.open(pose_path).convert("RGB").resize((512, 512))


    init_image = None
    if init_image_path:
        init_image = Image.open(init_image_path).convert("RGB").resize((512, 512))

    print(f"🎞️ Generating clip for prompt: {prompt[:80]}...")

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        num_inference_steps=step,
        guidance_scale=15,
        width=512,
        height=512,
        generator=generator,
        control_image=control_image,
        image=init_image  # Only effective if AnimateDiff supports img2img init
    )
    frames = output.frames[0]

    export_to_video(frames, output_path, fps=fps)
    print(f"✅ Saved animation: {output_path}")
