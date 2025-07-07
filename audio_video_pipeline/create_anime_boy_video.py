#!/usr/bin/env python3
"""
Create Anime Boy Video with Audio
Generate the complete anime boy story with synchronized audio
"""

import os
import sys
import time
from main_integration_pipeline import MainIntegrationPipeline, IntegrationConfig

def create_anime_boy_video():
    """Create the anime boy video with complete audio integration"""
    
    print("🎬 Creating Anime Boy Video with Audio Integration")
    print("=" * 60)
    
    # Your 10-line prompt
    anime_boy_prompts = [
        "Anime boy wearing a hoodie walks on a quiet street under a grey sky.",
        "Rain falls gently on anime boy as soft wind moves the hoodie.",
        "Anime boy stops at a glowing vending machine beside the road.",
        "Anime boy buys a warm canned coffee and holds the coffee with both hands.",
        "A small dog runs past anime boy, splashing water in anime style.",
        "Anime boy smiles and starts walking again through the calm street.",
        "Anime boy passes an anime bakery with warm yellow lights in the window.",
        "Anime boy pauses and looks inside the bakery as steam fogs the glass.",
        "Anime boy stands near a train crossing while red lights start flashing.",
        "Anime boy drinks the coffee slowly as the train moves fast through the rain."
    ]
    
    print(f"📝 Processing {len(anime_boy_prompts)} prompts:")
    for i, prompt in enumerate(anime_boy_prompts, 1):
        print(f"   {i:2d}. {prompt}")
    
    # Check if we have an existing video to use
    existing_video = os.path.join(os.path.dirname(__file__), '..', 'AnimateDiff', 'outputs', 'multi_clip', 'final_output_stitched.mp4')
    use_existing_video = os.path.exists(existing_video)

    if use_existing_video:
        print(f"✅ Found existing video: {os.path.basename(existing_video)}")
        print(f"   Using existing video instead of generating new one")
    else:
        print(f"⚠️ No existing video found, will generate new one")

    # Configuration for the pipeline
    config = IntegrationConfig(
        video_input_path=existing_video if use_existing_video else None,
        prompts=anime_boy_prompts,
        output_dir="anime_boy_outputs",
        narrator_gender="female",        # Female narrator for storytelling
        apply_lipsync=False,            # Skip lip-sync for faster processing
        enhance_prompts=True,           # Enable AI prompt enhancement
        generate_video=not use_existing_video  # Use existing video if available
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"   📁 Output directory: {config.output_dir}")
    print(f"   🎙️ Narrator gender: {config.narrator_gender}")
    print(f"   👄 Lip-sync: {'Enabled' if config.apply_lipsync else 'Disabled'}")
    print(f"   🤖 Prompt enhancement: {'Enabled' if config.enhance_prompts else 'Disabled'}")
    print(f"   🎬 Video generation: {'Enabled' if config.generate_video else 'Disabled'}")
    
    # Create the pipeline
    pipeline = MainIntegrationPipeline(config)
    
    start_time = time.time()
    
    try:
        print(f"\n🚀 Starting complete pipeline...")
        result = pipeline.process_complete_pipeline()
        
        total_time = time.time() - start_time
        
        if result.success:
            print(f"\n" + "=" * 60)
            print("🎉 ANIME BOY VIDEO CREATION SUCCESSFUL!")
            print("=" * 60)
            print(f"📹 Final video: {result.final_video_path}")
            print(f"📊 Video details:")
            print(f"   • Total prompts processed: {len(anime_boy_prompts)}")
            print(f"   • Audio layers created: {len(result.audio_layers)}")
            print(f"   • Narration layers: {len([l for l in result.audio_layers if l.layer_type == 'narration'])}")
            print(f"   • Dialogue layers: {len([l for l in result.audio_layers if l.layer_type == 'dialogue'])}")
            print(f"   • Processing time: {total_time:.2f}s")
            print(f"   • Pipeline processing time: {result.processing_time:.2f}s")
            
            # Calculate expected video duration (AnimateDiff: 32 frames @ 24 fps per clip)
            expected_duration = len(anime_boy_prompts) * (32 / 24)
            print(f"   • Expected video duration: {expected_duration:.2f}s")
            
            print(f"\n📁 Output location: {os.path.abspath(result.final_video_path)}")
            
            # Check if file exists and get size
            if os.path.exists(result.final_video_path):
                file_size = os.path.getsize(result.final_video_path) / (1024 * 1024)  # MB
                print(f"📊 File size: {file_size:.2f} MB")
            
            print(f"\n🎬 You can now play the video with synchronized audio!")
            
        else:
            print(f"\n" + "=" * 60)
            print("❌ ANIME BOY VIDEO CREATION FAILED!")
            print("=" * 60)
            print(f"Error: {result.error_message}")
            print(f"Processing time: {total_time:.2f}s")
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Process interrupted by user")
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n❌ Unexpected error: {str(e)}")
        print(f"Processing time: {total_time:.2f}s")
        
    finally:
        print(f"\n🧹 Cleaning up...")
        pipeline.cleanup()
        print(f"✅ Cleanup completed")

def check_prerequisites():
    """Check if all prerequisites are available"""
    print("🔍 Checking prerequisites...")
    
    # Check if AnimateDiff directory exists
    animatediff_dir = os.path.join(os.path.dirname(__file__), '..', 'AnimateDiff')
    if not os.path.exists(animatediff_dir):
        print(f"❌ AnimateDiff directory not found: {animatediff_dir}")
        return False
    
    # Check if multi_clip_generator.py exists
    multi_clip_file = os.path.join(animatediff_dir, 'multi_clip_generator.py')
    if not os.path.exists(multi_clip_file):
        print(f"❌ multi_clip_generator.py not found: {multi_clip_file}")
        return False
    
    # Check if SadTalker directory exists (for lip-sync, even if disabled)
    sadtalker_dir = os.path.join(os.path.dirname(__file__), '..', 'SadTalker')
    if not os.path.exists(sadtalker_dir):
        print(f"⚠️ SadTalker directory not found: {sadtalker_dir}")
        print("   Lip-sync functionality will be disabled")
    
    print("✅ Prerequisites check passed")
    return True

def main():
    """Main function"""
    print("🎌 Anime Boy Video Creator with Audio Integration")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please check your setup.")
        return
    
    # Create the video
    create_anime_boy_video()

if __name__ == "__main__":
    main()
