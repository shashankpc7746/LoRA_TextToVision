# text2img_generate_final.py

# Import the necessary pipeline classes
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import os
from typing import List, Union

# --- Agent Configuration ---
# This dictionary maps user-friendly style names to Hugging Face model IDs.
STYLE_MODELS = {
    "realistic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "cartoon": "prompthero/openjourney-v4",
    # "anime": "stablediffusionapi/anything-v5",
    "anime": "Linaqruf/animagine-xl",
    "fantasy": "Lykon/dreamshaper-8",
    "watercolor": "nitrosocke/Ghibli-Diffusion",
}

# This dictionary will cache loaded models in memory to prevent reloading during a single run.
loaded_pipelines = {}

def get_model_for_style(style: str) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    """
    Loads a pipeline for a given style. It now robustly checks if the model is an SDXL
    model by looking for "xl" in its name and loads the appropriate pipeline class.
    """
    if style in loaded_pipelines:
        print(f"✅ Using cached model for '{style}' style.")
        return loaded_pipelines[style]

    model_id = STYLE_MODELS.get(style)
    if not model_id:
        raise ValueError(f"❌ Unknown style '{style}'. Choose from: {list(STYLE_MODELS.keys())}")
    
    print(f"🔄 Loading model for '{style}' style: {model_id}...")
    
    # ✅ --- THE ROBUST FIX ---
    # Instead of a separate list, we check if "xl" is in the model's name.
    is_xl = "xl" in model_id.lower()
    PipelineClass = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
    
    print(f"   Using Pipeline: {PipelineClass.__name__}")

    cache_dir = "./huggingface_cache"
    
    # Load the model using the determined pipeline class
    pipe = PipelineClass.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16" if is_xl else None, # Use fp16 variant for XL models
        cache_dir=cache_dir
    ).to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled.")
    except Exception as e:
        print(f"⚠️ Could not enable xformers. Running without it. Error: {e}")
    
    loaded_pipelines[style] = pipe
    return pipe

def generate_images_from_prompts(prompt_input: Union[str, List[str]], style: str = "realistic", seed: int = 1234):
    """
    Generates a sequence of images from either a single paragraph string or a list of strings.
    """
    print(f"\n--- 🚀 Starting Generation for Style: {style.upper()} ---")
    pipe = get_model_for_style(style)
    
    # Process the input prompts
    if isinstance(prompt_input, list):
        prompts = [p.strip() for p in prompt_input if p.strip()]
    elif isinstance(prompt_input, str):
        prompts = [s.strip() for s in prompt_input.strip().split('.') if s.strip()]
    else:
        raise TypeError("Input must be a paragraph string or a list of sentence strings.")
    
    # Define quality and negative prompts based on the style
    if "anime" in style.lower() or "xl" in pipe.config._name_or_path:
        quality_prompt = "masterpiece, best quality"
        negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    else:
        quality_prompt = "cinematic, beautiful, high detail"
        negative_prompt = "blurry, out of frame, low quality, distorted, extra limbs, watermark, signature, ugly"

    output_dir = f"phase2_outputs/{style}"
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
    
    story_paragraph = """
    A majestic fox with a fiery tail wandered into an ancient, enchanted forest.
    He discovered a glowing, magical book resting on a mossy pedestal.
    """
    print("--- Running example with a paragraph string (Anime XL) ---")
    generate_images_from_prompts(story_paragraph, style="anime", seed=123455)

    # # --- Run Example 2 (Standard SD 1.5 model) ---
    # prompt_list = [
    #     "A cute robot building a sandcastle on a beach.",
    #     "The robot looks up as a friendly crab approaches.",
    # ]
    # print("\n--- Running example with a list of prompts (Fantasy) ---")
    # generate_images_from_prompts(prompt_list, style="anime", seed=12345)
