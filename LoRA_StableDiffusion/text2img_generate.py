# text2img_generate_final.py

# Import the necessary pipeline classes
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import os
from typing import List, Union

# --- Agent Configuration ---
STYLE_MODELS = {
    "realistic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "cartoon": "prompthero/openjourney-v4",
    # "anime": "stablediffusionapi/anything-v5",  # optional choice
    "anime": "Linaqruf/animagine-xl",
    "fantasy": "Lykon/dreamshaper-8",
    "watercolor": "nitrosocke/Ghibli-Diffusion",
}

loaded_pipelines = {}

def get_model_for_style(style: str) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    if style in loaded_pipelines:
        print(f"✅ Using cached model for '{style}' style.")
        return loaded_pipelines[style]

    model_id = STYLE_MODELS.get(style)
    if not model_id:
        raise ValueError(f"❌ Unknown style '{style}'. Choose from: {list(STYLE_MODELS.keys())}")
    
    print(f"🔄 Loading model for '{style}' style: {model_id}...")

    is_xl = "xl" in model_id.lower()
    PipelineClass = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

    print(f"   Using Pipeline: {PipelineClass.__name__}")

    cache_dir = "./huggingface_cache"

    pipe = PipelineClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16" if is_xl else None,
        cache_dir=cache_dir
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled.")
    except Exception as e:
        print(f"⚠️ Could not enable xformers. Running without it. Error: {e}")
    
    loaded_pipelines[style] = pipe
    return pipe

def get_next_scene_number(base_dir="VideoMaker") -> int:
    existing = [d for d in os.listdir(base_dir) if d.startswith("frames_scene_")]
    scene_nums = [int(d.split("_")[-1]) for d in existing if d.split("_")[-1].isdigit()]
    return max(scene_nums) + 1 if scene_nums else 1

def generate_images_from_prompts(prompt_input: Union[str, List[str]], style: str = "realistic", seed: int = 1234, scene_number: int = None):
    """
    Generates a sequence of images from either a single paragraph string or a list of strings.
    Saves them in VideoMaker/frames_scene_{scene_number}/
    """
    print(f"\n--- 🚀 Starting Generation for Style: {style.upper()} ---")
    pipe = get_model_for_style(style)

    if isinstance(prompt_input, list):
        prompts = [p.strip() for p in prompt_input if p.strip()]
    elif isinstance(prompt_input, str):
        prompts = [s.strip() for s in prompt_input.strip().split('.') if s.strip()]
    else:
        raise TypeError("Input must be a paragraph string or a list of sentence strings.")

    if "anime" in style.lower() or "xl" in pipe.config._name_or_path:
        quality_prompt = "masterpiece, best quality"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    else:
        quality_prompt = "cinematic, beautiful, high detail"
        negative_prompt = "blurry, out of frame, low quality, distorted, extra limbs, watermark, signature, ugly"

    base_dir = "VideoMaker"
    os.makedirs(base_dir, exist_ok=True)

    if scene_number is None:
        scene_number = get_next_scene_number(base_dir)

    output_dir = os.path.join(base_dir, f"frames_scene_{scene_number}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Images will be saved to: {output_dir}")
    print(f"🌱 Using generation seed: {seed}")

    generator = torch.Generator("cuda").manual_seed(seed)

    for i, prompt in enumerate(prompts):
        full_prompt = f"{prompt.strip()}, {quality_prompt}"

        print(f"\n🖼️  Generating image {i+1}/{len(prompts)}...")
        print(f"   Prompt: {full_prompt}")

        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=7.5,
            num_inference_steps=30,
            generator=generator,
        ).images[0]

        image_path = os.path.join(output_dir, f"frame_{i+1:03d}.png")
        image.save(image_path)
        print(f"✅ Saved to: {image_path}")

    print(f"\n🎉 Completed generation for style: {style}. All images saved in {output_dir}")


# Main execution block
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Run Example 1 (SDXL Anime model) ---
    
    # story_paragraph = """
    # A bear cub wakes up in a treehouse.  
    # The bear cub slides down a vine.  
    # A butterfly lands on the bear cub’s nose.  
    # The bear cub finds glowing fish in a pond.  
    # The bear cub joins a picnic with other animals.
    # """
    
    story_paragraph = """
    
    A young wizard stands near his cottage.
    The wizard opens a glowing spellbook.
    The wizard waves his wand at a nearby tree.
    The wizard watches as the tree starts floating.

    """
    print("--- Running example with a paragraph~ ---")
    generate_images_from_prompts(story_paragraph, style="anime", seed=12345)