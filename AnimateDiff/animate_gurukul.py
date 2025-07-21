import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import export_to_video
from PIL import Image

# ===== CENTRALIZED VIDEO SETTINGS =====
# Change FPS here and it will be used everywhere
device = "cuda"
dtype = torch.float16
step = 25
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model = "stabilityai/sd-vae-ft-mse"
num_frames = 32
fps = 12  # ← SINGLE FPS SETTING FOR ALL VIDEO GENERATION

# Load Motion Adapter
adapter = MotionAdapter.from_pretrained(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype=dtype
).to(device)

# Load AnimateDiff pipeline
pipe = AnimateDiffPipeline.from_pretrained(
    base_model,
    motion_adapter=adapter,
    torch_dtype=dtype
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
