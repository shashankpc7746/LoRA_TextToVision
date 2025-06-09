# text2img_generate.py (No NLTK version)

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

# --- Agent Configuration ---
# ✅ 1. Define a mapping of user-friendly style names to Hugging Face model IDs.
STYLE_MODELS = {
    "realistic": "SG161222/Realistic_Vision_V5.1_noVAE",
    "cartoon": "prompthero/openjourney-v4", # A versatile cartoon/fantasy model
    "anime": "andite/anything-v4.0", # A classic high-quality anime model
    "fantasy": "Lykon/dreamshaper-8", # Excellent for fantasy and illustrative styles
}

# ✅ 2. A function to load the correct model pipeline based on the selected style
def load_model_for_style(style):
    """Loads the appropriate Stable Diffusion pipeline for the given style."""
    model_id = STYLE_MODELS.get(style)
    if not model_id:
        raise ValueError(f"❌ Unknown style '{style}'. Choose from: {list(STYLE_MODELS.keys())}")
    
    print(f"🔄 Loading model for '{style}' style: {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None 
    ).to("cuda")

    # ✅ Enable xformers memory-efficient attention for a significant speed-up and lower VRAM usage.
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✅ xformers memory-efficient attention enabled.")
    except Exception as e:
        print(f"⚠️ Could not enable xformers. Running without it. Error: {e}")
        
    return pipe

# ✅ 3. The main function to generate a sequence of images from a story
def generate_images_from_story(story_text, style="realistic"):
    """
    Takes a block of text (a story) and a style, then generates a sequence of images,
    one for each sentence.
    """
    print(f"\n--- 🚀 Starting Generation for Style: {style.upper()} ---")
    pipe = load_model_for_style(style)
    
    # ❌ Removed NLTK.
    # ✅ Use simple string splitting to break the story into sentences.
    # This splits the text by periods and removes any empty strings or extra whitespace.
    prompts = [sentence.strip() for sentence in story_text.strip().split('.') if sentence.strip()]
    
    # A general negative prompt to improve quality across all models
    negative_prompt = "blurry, out of frame, low quality, distorted, extra limbs, watermark, signature, ugly"

    # Create a dedicated output directory for the style
    output_dir = f"phase2_outputs/{style}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Images will be saved to: {output_dir}")

    # Generate one image for each sentence
    for i, prompt in enumerate(prompts):
        # We can add style-specific keywords to the prompt to enhance the effect
        full_prompt = f"{prompt}, {style} style, masterpiece, best quality, cinematic"
        
        print(f"\n🖼️  Generating image {i+1}/{len(prompts)}...")
        print(f"   Prompt: {full_prompt}")

        # For reproducibility, use a generator with a fixed seed
        generator = torch.Generator("cuda").manual_seed(1234)
        
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            guidance_scale=8.5,
            num_inference_steps=50,
            generator=generator,
        ).images[0]

        image_path = os.path.join(output_dir, f"frame_{i+1:03d}.png") # Use padding for better sorting (e.g., frame_001.png)
        image.save(image_path)
        print(f"✅ Saved to: {image_path}")

    print(f"\n🎉 Completed generation for style: {style}. All images saved in {output_dir}")


# ✅ 4. Example Usage
if __name__ == "__main__":
    # A sample story to generate a video from
    story = """
    A majestic fox with a fiery tail wandered into an ancient, enchanted forest.
    He discovered a glowing, magical book resting on a mossy pedestal.
    As the fox curiously touched the book, a swirling portal of stars opened above.
    A celestial dragon, made of constellations, emerged from the portal, offering the fox a cosmic crown.
    """

    # 🔁 Change this to try different styles: "realistic", "cartoon", "anime", "fantasy"
    selected_style = "fantasy" 

    generate_images_from_story(story, selected_style)
