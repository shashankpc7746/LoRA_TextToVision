"""
Generate Diverse Educational Training Dataset for Gurukul LoRA
Creates 200-500 AI-generated educational images using SDXL across all domains
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

print("Starting imports (this may take 1-2 minutes)...")

import torch
print("✓ torch")

from pathlib import Path
import json
from tqdm import tqdm
import argparse
from datetime import datetime
print("✓ basic libraries")

print("Loading diffusers...", end="", flush=True)
from diffusers import DiffusionPipeline
print(" ✓")


class EducationalDatasetGenerator:
    """Generate diverse educational images for general-purpose learning content"""
    
    def __init__(self, output_dir="datasets/gurukul_keyframes", device="cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Educational prompt templates across ALL domains
        self.prompt_templates = {
            "stem_math": [
                "clean educational diagram of mathematical equation on whiteboard, professional photo, high quality",
                "geometry shapes and formulas on blackboard, educational setting, clear lighting",
                "algebra equations being solved step by step, neat handwriting, educational",
                "calculus graphs and derivatives on digital screen, modern classroom, professional",
                "statistics chart with data visualization, clean infographic style, educational",
                "trigonometry unit circle diagram, colorful educational poster, clear labels",
                "mathematical proof on whiteboard, university lecture, professional lighting",
                "number theory concepts visualization, educational infographic, modern design",
                "linear algebra matrices and vectors, digital presentation, clean layout",
                "probability tree diagram, educational illustration, clear and organized",
            ],
            
            "stem_physics": [
                "physics laboratory experiment setup with equipment, professional photo, bright lighting",
                "Newton's laws demonstration with objects, educational setting, clear composition",
                "electricity circuit diagram with components, educational poster, labeled clearly",
                "optics light refraction experiment, laboratory setting, professional photo",
                "mechanics force diagrams on whiteboard, educational, neat and clear",
                "thermodynamics heat transfer visualization, educational diagram, modern style",
                "quantum mechanics wave-particle duality illustration, educational, artistic",
                "electromagnetism field lines diagram, educational poster, colorful",
                "astronomy solar system model, educational display, professional lighting",
                "classical mechanics pendulum experiment, laboratory photo, high quality",
            ],
            
            "stem_chemistry": [
                "chemistry molecular structure 3D visualization, educational, colorful atoms",
                "periodic table of elements poster, modern design, educational, clear labels",
                "chemical reaction equation on whiteboard, educational setting, neat writing",
                "laboratory glassware with colored liquids, professional photo, bright lighting",
                "organic chemistry benzene ring structure, educational diagram, clear",
                "chemical bonding ionic and covalent illustration, educational poster, modern",
                "pH scale color chart, educational infographic, vibrant colors",
                "stoichiometry calculation on digital screen, educational, step by step",
                "electrochemistry battery diagram, educational illustration, labeled",
                "biochemistry protein structure model, educational 3D visualization, detailed",
            ],
            
            "stem_biology": [
                "biology cell diagram with organelles labeled, educational poster, colorful",
                "human anatomy heart structure illustration, medical education, detailed",
                "DNA double helix structure model, educational 3D visualization, modern",
                "plant cell vs animal cell comparison, educational diagram, side by side",
                "ecosystem food chain illustration, educational poster, nature theme",
                "microscope view of cells, scientific photo, high magnification, clear",
                "human skeleton anatomy poster, medical education, labeled bones",
                "photosynthesis process diagram, educational illustration, step by step",
                "genetics Punnett square on whiteboard, educational, clear handwriting",
                "neuron cell structure diagram, medical education, detailed labels",
            ],
            
            "stem_computer": [
                "computer programming code on screen Python, modern workspace, professional photo",
                "algorithm flowchart diagram, educational poster, clean design, arrows",
                "data structures visualization arrays and lists, educational diagram, colorful",
                "software development IDE interface, coding tutorial, realistic screenshot",
                "network topology diagram computers connected, educational illustration, clear",
                "database schema with tables and relationships, educational diagram, organized",
                "artificial intelligence neural network visualization, educational, modern",
                "web development HTML CSS code editor, professional workspace photo",
                "binary code and digital data visualization, educational, matrix style",
                "cybersecurity encryption concept illustration, educational, lock and key theme",
            ],
            
            "humanities_history": [
                "world history timeline infographic, educational poster, clean design, dates",
                "ancient civilization map with empires, educational, vintage style map",
                "historical figures portrait gallery, educational collage, professional",
                "industrial revolution machines and factories, educational illustration, detailed",
                "world war historical map with battle locations, educational, clear markings",
                "renaissance art and architecture examples, educational poster, classical",
                "medieval castle and knight illustration, educational, historical accuracy",
                "ancient Egypt pyramids and hieroglyphics, educational poster, golden theme",
                "Roman Empire architecture and roads map, educational illustration, detailed",
                "Cold War political map with divisions, educational, clear boundaries",
            ],
            
            "humanities_geography": [
                "world map with continents and oceans labeled, educational poster, colorful",
                "topographic map with elevation contours, educational, terrain colors",
                "climate zones of Earth diagram, educational poster, temperature gradient",
                "tectonic plates and earthquake zones map, educational, geological",
                "river systems and watershed illustration, educational diagram, blue theme",
                "political map of countries and capitals, educational poster, clear labels",
                "ecosystem biomes world distribution, educational map, nature colors",
                "ocean currents global circulation map, educational, flowing arrows",
                "mountain ranges and peaks illustration, educational, 3D relief style",
                "urban city planning and zoning map, educational diagram, organized layout",
            ],
            
            "humanities_literature": [
                "open classic book with pages visible, study desk setting, warm lighting",
                "library bookshelf with organized books, educational setting, cozy atmosphere",
                "poetry verses handwritten on paper, artistic, elegant handwriting",
                "Shakespeare quote on vintage parchment, educational poster, classic style",
                "writing process brainstorm to final draft, educational diagram, organized",
                "literary devices examples poster, educational, colorful text boxes",
                "story structure plot diagram, educational illustration, mountain shape",
                "grammar rules and punctuation guide, educational poster, clear examples",
                "famous authors portrait collection, educational collage, vintage style",
                "book analysis mind map with themes, educational diagram, connected ideas",
            ],
            
            "languages": [
                "alphabet letters A to Z poster, educational, colorful and bold fonts",
                "language learning flashcards spread out, educational, vocabulary words",
                "foreign language phrases with translations, educational poster, two columns",
                "phonetics and pronunciation guide chart, educational, IPA symbols",
                "grammar sentence structure diagram, educational, tree diagram style",
                "vocabulary word wall with images, educational classroom, colorful cards",
                "language conjugation table verbs, educational chart, organized grid",
                "bilingual dictionary page spread, educational, two languages side by side",
                "handwriting practice worksheet letters, educational, dotted guidelines",
                "language family tree diagram, educational, branches showing relationships",
            ],
            
            "arts_music": [
                "musical instruments collection arranged professionally, studio lighting, high quality",
                "music sheet with notes and clefs, educational, clear staff lines",
                "piano keyboard diagram with note labels, educational poster, black and white keys",
                "guitar chord chart for beginners, educational diagram, finger positions",
                "music theory circle of fifths illustration, educational, colorful wheel",
                "orchestra seating arrangement diagram, educational, instrument sections",
                "musical notation symbols guide, educational poster, various notes and signs",
                "sound waves and frequency visualization, educational diagram, colorful waves",
                "music composition software interface, digital workspace, professional",
                "rhythm and tempo notation examples, educational chart, beat patterns",
            ],
            
            "arts_visual": [
                "art supplies brushes paints and canvas, creative workspace, bright lighting",
                "color wheel theory diagram primary and secondary, educational poster, vibrant",
                "drawing perspective one point and two point, educational tutorial, sketch style",
                "painting techniques demonstration brushstroke examples, educational, various styles",
                "sculpture tools and clay workspace, art studio, professional photo",
                "art history movements timeline, educational infographic, various art styles",
                "photography composition rules of thirds, educational diagram, grid overlay",
                "digital art tablet and stylus workspace, modern creative setup, professional",
                "origami folding instructions step by step, educational diagram, paper craft",
                "calligraphy letterforms and pen strokes, educational poster, elegant writing",
            ],
            
            "professional_business": [
                "business presentation slide on screen graphs and charts, modern office, professional",
                "financial charts stock market trends, educational diagram, line graphs",
                "marketing strategy mind map, educational illustration, connected concepts",
                "business meeting conference room, professional setting, bright lighting",
                "organizational hierarchy chart, educational diagram, tree structure",
                "project management Gantt chart timeline, educational, colored bars",
                "economics supply and demand curves, educational graph, intersection point",
                "accounting balance sheet example, educational document, organized columns",
                "business plan document template, educational, professional layout",
                "entrepreneurship startup process flowchart, educational diagram, step by step",
            ],
            
            "professional_technology": [
                "engineering blueprint technical drawing, professional, detailed measurements",
                "3D CAD model on computer screen, engineering workspace, modern software",
                "robotics components and circuits board, educational, technology theme",
                "mechanical engineering gears and mechanisms, educational diagram, cross section",
                "electrical engineering circuit board close-up, professional photo, components visible",
                "civil engineering bridge design diagram, educational, structural analysis",
                "aerospace engineering aircraft blueprint, educational, technical drawing",
                "automotive engineering engine diagram, educational, exploded view",
                "chemical engineering process flow diagram, educational, equipment and pipes",
                "industrial automation factory robots, educational photo, modern facility",
            ],
            
            "professional_medical": [
                "medical anatomy human body systems poster, educational, detailed illustration",
                "healthcare diagnosis procedure chart, educational diagram, flowchart style",
                "first aid emergency steps illustration, educational poster, numbered steps",
                "medical equipment stethoscope and tools, professional photo, clinical setting",
                "pharmaceutical medicine bottles and pills, educational photo, labeled",
                "surgical instruments arranged on tray, medical education, sterile setting",
                "radiology X-ray image of bones, medical education, clear contrast",
                "nutrition food pyramid healthy eating, educational diagram, colorful sections",
                "medical chart with patient vitals, educational document, organized data",
                "microscope laboratory medical research, professional photo, scientific setting",
            ],
            
            "general_classroom": [
                "modern classroom with digital whiteboard students, bright lighting, professional photo",
                "university lecture hall with projector screen, educational setting, wide angle",
                "online learning video conference screen, modern workspace, professional setup",
                "student studying with laptop and textbooks, desk setup, natural lighting",
                "teacher explaining at whiteboard students listening, classroom, engaging scene",
                "library study area with desks and computers, educational space, quiet atmosphere",
                "collaborative learning group discussion, students working together, modern space",
                "science laboratory classroom with equipment, educational setting, bright and clean",
                "art classroom with easels and supplies, creative space, colorful environment",
                "computer lab students at workstations, educational facility, modern equipment",
            ],
            
            "general_digital": [
                "educational technology tablet showing learning app, professional photo, clear screen",
                "e-learning platform interface on laptop, modern design, user friendly",
                "virtual reality VR headset for education, student using, futuristic",
                "interactive whiteboard smartboard in classroom, modern technology, professional",
                "educational mobile app interface design, smartphone screen, clean UI",
                "digital learning management system dashboard, computer screen, organized layout",
                "online course video lesson recording setup, professional studio, lighting and camera",
                "educational podcast recording equipment, microphone and laptop, studio setting",
                "augmented reality AR educational content, tablet overlay, innovative technology",
                "cloud storage educational resources icons, digital illustration, organized folders",
            ],
        }
        
    def get_all_prompts(self, num_images=300):
        """Get balanced prompts across all categories"""
        all_prompts = []
        
        # Flatten all prompts
        for category, prompts in self.prompt_templates.items():
            all_prompts.extend([(p, category) for p in prompts])
        
        # Calculate how many times to repeat
        repeats = (num_images // len(all_prompts)) + 1
        
        # Repeat and shuffle
        expanded_prompts = all_prompts * repeats
        
        # Shuffle for variety
        import random
        random.seed(42)  # Deterministic for reproducibility
        random.shuffle(expanded_prompts)
        
        # Return exact number needed
        return expanded_prompts[:num_images]
    
    def generate_dataset(self, num_images=300, batch_size=1):
        """
        Generate complete training dataset
        
        Args:
            num_images: Number of images to generate (default: 300)
            batch_size: Batch size for generation (default: 1 for stability)
        """
        print(f"\n{'='*70}")
        print(f"EDUCATIONAL DATASET GENERATOR - Gurukul Training")
        print(f"{'='*70}\n")
        
        print(f"📊 Configuration:")
        print(f"   Total Images: {num_images}")
        print(f"   Categories: {len(self.prompt_templates)}")
        print(f"   Output: {self.output_dir}")
        print(f"   Device: {self.device}\n")
        
        # Load SDXL pipeline
        print("🔧 Loading SDXL pipeline...")
        try:
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                local_files_only=True
            ).to(self.device)
            
            pipe.enable_attention_slicing()
            print("✅ SDXL pipeline loaded successfully\n")
            
        except Exception as e:
            print(f"❌ Failed to load SDXL pipeline: {e}")
            print("\nTrying without local_files_only...")
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            ).to(self.device)
            pipe.enable_attention_slicing()
            print("✅ SDXL pipeline loaded successfully\n")
        
        # Get prompts
        prompts_with_categories = self.get_all_prompts(num_images)
        
        # Generate images
        print(f"🎨 Generating {num_images} educational images...\n")
        
        captions = {}
        category_counts = {}
        
        for i, (prompt, category) in enumerate(tqdm(prompts_with_categories, desc="Generating")):
            try:
                # Generate image
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=1024,
                    width=1024
                ).images[0]
                
                # Save image
                img_name = f"keyframe_{i+1:04d}.png"
                img_path = self.output_dir / img_name
                image.save(img_path)
                
                # Store caption (clean prompt - remove technical descriptions)
                clean_caption = prompt.split(',')[0]  # Take main subject only
                captions[img_name] = clean_caption
                
                # Count categories
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Progress update every 50 images
                if (i + 1) % 50 == 0:
                    print(f"\n✅ Generated {i+1}/{num_images} images")
                    
            except Exception as e:
                print(f"\n⚠️ Failed to generate image {i+1}: {e}")
                continue
        
        # Save captions
        caption_file = self.output_dir / "captions.json"
        with open(caption_file, 'w') as f:
            json.dump(captions, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ DATASET GENERATION COMPLETE")
        print(f"{'='*70}\n")
        
        print(f"📊 Summary:")
        print(f"   Total Images: {len(captions)}")
        print(f"   Output Directory: {self.output_dir}")
        print(f"   Captions File: {caption_file}\n")
        
        print(f"📈 Category Distribution:")
        for category, count in sorted(category_counts.items()):
            print(f"   {category:25s}: {count:3d} images")
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Review generated images in: {self.output_dir}")
        print(f"   2. Start 100-epoch training: python train_optimized.py")
        print(f"   3. Expected training time: ~37.5 hours\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate diverse educational training dataset")
    parser.add_argument("--num_images", type=int, default=300, 
                       help="Number of images to generate (default: 300)")
    parser.add_argument("--output_dir", type=str, default="datasets/gurukul_keyframes",
                       help="Output directory for dataset")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation (default: 1)")
    
    args = parser.parse_args()
    
    # Create generator
    generator = EducationalDatasetGenerator(
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_images=args.num_images,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
