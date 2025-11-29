"""
Download remaining images to complete 500-image dataset
"""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))
from download_production_dataset import EducationalDatasetDownloader

def main():
    output_dir = Path(r'c:\Shashank\LoRA_TextToVision\datasets\gurukul_keyframes')
    captions_file = output_dir / 'captions.json'
    
    # Load existing captions
    with open(captions_file, 'r') as f:
        existing_captions = json.load(f)
    
    print(f'\n📊 Current dataset: {len(existing_captions)} images')
    
    # Count by source
    pexels_existing = len([k for k in existing_captions.keys() if k.startswith('pexels_')])
    wiki_existing = len([k for k in existing_captions.keys() if k.startswith('wiki_')])
    openimg_existing = len([k for k in existing_captions.keys() if k.startswith('openimg_')])
    
    print(f'  - Pexels: {pexels_existing}/200 (need {200-pexels_existing} more)')
    print(f'  - WikiMedia: {wiki_existing}/100 ✅')
    print(f'  - Open Images: {openimg_existing}/200 (need {200-openimg_existing} more)')
    
    # Create downloader
    downloader = EducationalDatasetDownloader(output_dir=str(output_dir), test_mode=False)
    downloader.captions = existing_captions.copy()
    downloader.downloaded_images = list(existing_captions.keys())
    
    # Set limits for remaining images
    pexels_needed = max(0, 200 - pexels_existing)
    openimg_needed = max(0, 200 - openimg_existing)
    
    downloader.limits['pexels'] = pexels_needed
    downloader.limits['wikimedia'] = 0  # Already complete
    downloader.limits['open_images'] = openimg_needed
    
    print(f'\n🚀 Starting download of remaining images...\n')
    
    # Download Pexels
    if pexels_needed > 0:
        print(f'📥 Downloading {pexels_needed} more Pexels images...')
        pexels_api_key = os.getenv('PEXELS_API_KEY')
        if not pexels_api_key:
            print('⚠️  PEXELS_API_KEY not found in environment variables')
            pexels_count = 0
        else:
            pexels_count = downloader.download_pexels(api_key=pexels_api_key)
            print(f'✅ Downloaded {pexels_count} Pexels images\n')
    else:
        pexels_count = 0
        print('✅ Pexels quota already met\n')
    
    # Download Open Images
    if openimg_needed > 0:
        print(f'📥 Downloading {openimg_needed} more Open Images...')
        openimg_count = downloader.download_open_images()
        print(f'✅ Downloaded {openimg_count} Open Images\n')
    else:
        openimg_count = 0
        print('✅ Open Images quota already met\n')
    
    # Save updated captions
    downloader.generate_captions_file()
    
    # Final summary
    total_new = pexels_count + openimg_count
    total_final = len(downloader.captions)
    
    print(f'\n{"="*70}')
    print('📊 FINAL SUMMARY')
    print(f'{"="*70}\n')
    print(f'Previously: {len(existing_captions)} images')
    print(f'Downloaded: {total_new} new images')
    print(f'Total now: {total_final} images')
    print(f'Target: 500 images')
    
    if total_final >= 500:
        print(f'\n✅ TARGET REACHED! Dataset is complete!\n')
    else:
        print(f'\n⚠️  Still need {500 - total_final} more images\n')

if __name__ == "__main__":
    main()
