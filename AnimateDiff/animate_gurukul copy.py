import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import export_to_video

device = "cuda"; dtype = torch.float16
step = 25
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model = "stabilityai/sd-vae-ft-mse"

# prompt = "Young wizard walking  an ancient enchanted forest at golden hour, staff in hand, soft fog swirling, glowing magical particles around him. Warm cinematic lighting, ultra-detailed 4K, fantasy anime style."
# prompt = "A powerful wizard with a lthroughong flowing robe and glowing staff, casting a fire spell with magical runes swirling around, intense lighting, sparks and particles in the air, fantasy background, dramatic pose, ultra-detailed, cinematic, 4K"
# prompt = "close-up anime scene of two dragons clashing mid-air — one breathes glowing amber fire, the other unleashes indigo dark flames. Stormy dawn sky with heavy rain clouds. Vast desert with giant sand dunes below. A few god-like beings watch the battle. Cinematic lighting, ultra-detailed, 4K, epic fantasy style."
# prompt = "A monkey sitting on a tree eating apple, natural lighting, high details, ultra-sharp focus,low motion blur, smooth movement, 4K"
# prompt = "A highly realistic cinematic scene of a massive, breathtaking island floating in outer space. The island is covered in dense, alien jungle with bioluminescent plants, glowing crystal formations, and mist rising from strange rivers. The stars, nebula clouds, and a distant ringed gas giant fill the sky, casting a soft ambient glow. 4K"
# prompt = "A young girl stands in an abandoned Victorian mansion at night, holding a flickering candle. The wallpaper peels, revealing faces that shift and whisper. As thunder crashes outside, a shadowy figure appears behind her in the mirror, but when she turns—no one's there. Slowly, the walls begin to bleed ink, and the chandelier swings violently above her."
# prompt = "High school student with glowing blue eyes summons a massive phoenix on a school rooftop at sunset. Dynamic 360° camera spin as cherry blossom petals swirl in the sky. His rival launches a sword strike with glowing energy trails. The phoenix screeches and dives, engulfing the rooftop in golden flames before vanishing. Epic anime-style, cinematic lighting, ultra-detailed, 4K."
# prompt = "In 16th-century Mughal India, an emperor rides an elephant through a grand fort gate, draped in silk banners and guarded by soldiers in ornate armor. A royal procession follows with dancers, musicians, and nobles. As the emperor looks out from the fort balcony, fireworks begin to burst over the palace during a nighttime festival."
# prompt = "A team of astronauts explores a derelict alien spacecraft orbiting a collapsing star. The scene cuts between zero-gravity exploration inside the eerie corridors and flashbacks of their loved ones on Earth. Suddenly, the ship's time dilation field activates, causing the walls to ripple and the lights to flicker in sync with the astronaut’s heartbeat."
prompt = "A man climbs an endless staircase floating in a void, where clocks melt on the steps and giant eyes blink in the sky. With each step, gravity shifts—fish swim through the air, and buildings bend like rubber. As he reaches the top, the staircase folds into a Mobius strip, trapping him in a loop of his own dreams."

negative_prompt = "blurry, distorted, ghost eyes, unnatural skin, motion-blur, scene flicker , extra limbs, low quality"

num_frames = 32; fps = 8
output_path = "./outputs/monkey_eating_highquality.mp4"
generator = torch.Generator(device).manual_seed(401)

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

# (optional) memory optimizations
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# Generate
print("🎬 Generating animation...")
output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=num_frames,
    num_inference_steps=step,
    guidance_scale=15,
    width=512,
    height=512,
    generator=generator
)
frames = output.frames[0]

# Save
export_to_video(frames, output_path, fps=fps)
print(f"✅ Saved animation: {output_path}")