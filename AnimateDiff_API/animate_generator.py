# animate_generator.py
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler, AutoencoderKL
from diffusers.utils import export_to_video
from datetime import datetime
import os

device = "cuda" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# Static load (once)
base_model = "SG161222/Realistic_Vision_V5.1_noVAE"
vae_model = "stabilityai/sd-vae-ft-mse"
motion_adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=dtype).to(device)

pipe = AnimateDiffPipeline.from_pretrained(
    base_model,
    motion_adapter=motion_adapter,
    torch_dtype=dtype
).to(device)

pipe.vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype).to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

def generate_video(prompt, negative_prompt=None, num_frames=32, steps=25, guidance_scale=15, seed=333, fps=8):
    generator = torch.Generator(device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=512,
        height=512,
        generator=generator
    )
    frames = output.frames[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"animation_{timestamp}.mp4"
    output_path = os.path.join("outputs", filename)
    export_to_video(frames, output_path, fps=fps)

    return output_path
