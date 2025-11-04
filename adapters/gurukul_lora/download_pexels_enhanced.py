"""
Download remaining Pexels images using more diverse keywords and pagination
"""
import sys
from pathlib import Path
import json
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

def download_more_pexels(api_key, target_count, existing_count, output_dir):
    """Download more Pexels images using diverse keywords and pagination"""
    
    # Load existing captions to avoid duplicates
    captions_file = output_dir / 'captions.json'
    with open(captions_file, 'r') as f:
        existing_captions = json.load(f)
    
    existing_ids = set([k for k in existing_captions.keys() if k.startswith('pexels_')])
    print(f"Found {len(existing_ids)} existing Pexels images")
    
    # More diverse keywords - including generic terms
    keywords = [
        # Generic education terms
        "university lecture hall", "library study room", "laboratory equipment",
        "school blackboard", "student notebook", "textbook pages",
        "education supplies", "academic workspace", "scientific microscope",
        "math equations board", "reading desk", "writing paper",
        "group study session", "professor teaching", "exam preparation",
        "graduation ceremony", "school supplies", "lecture notes",
        "research papers", "academic books", "calculator",
        # Subject-specific
        "algebra mathematics", "geometry shapes", "statistics charts",
        "chemistry beakers", "biology cells", "physics formulas",
        "historical documents", "world map globe", "literature novels",
        "musical notation", "art painting canvas", "piano keyboard",
        "business charts", "engineering blueprints", "medical anatomy",
        # Learning contexts
        "distance learning", "e-learning computer", "homework desk",
        "study lamp books", "educational poster", "classroom board",
        "seminar presentation", "workshop training", "academic conference",
        "student laptop", "online course", "virtual classroom",
        # Different angles
        "education background", "learning concept", "knowledge books",
        "wisdom library", "study motivation", "academic success"
    ]
    
    api_url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    
    downloaded = 0
    downloaded_images = []
    new_captions = existing_captions.copy()
    
    print(f"\n🎯 Target: Download {target_count} more Pexels images")
    print(f"📚 Using {len(keywords)} diverse keywords\n")
    
    for keyword in tqdm(keywords, desc="Searching Pexels"):
        if downloaded >= target_count:
            break
        
        # Try multiple pages for each keyword
        for page in range(1, 4):  # Try up to 3 pages
            if downloaded >= target_count:
                break
            
            try:
                params = {
                    "query": keyword,
                    "per_page": 10,  # Get more per request
                    "page": page
                }
                
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'photos' not in data or not data['photos']:
                    break  # No more results for this keyword
                
                for photo in data['photos']:
                    if downloaded >= target_count:
                        break
                    
                    # Create unique filename based on Pexels ID
                    pexels_id = photo['id']
                    filename = f"pexels_{pexels_id}.png"
                    
                    # Skip if already downloaded
                    if filename in existing_ids:
                        continue
                    
                    # Download image
                    try:
                        img_url = photo['src']['large']  # Get large version
                        img_response = requests.get(img_url, timeout=30, stream=True)
                        img_response.raise_for_status()
                        
                        # Process image
                        img = Image.open(BytesIO(img_response.content))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                        
                        # Save
                        filepath = output_dir / filename
                        img.save(filepath, 'PNG', quality=95)
                        
                        # Add to captions
                        caption = f"Educational image: {photo.get('alt', keyword)}"
                        new_captions[filename] = caption
                        existing_ids.add(filename)
                        downloaded_images.append(filename)
                        downloaded += 1
                        
                    except Exception as e:
                        continue
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                continue
        
        if downloaded > 0 and downloaded % 20 == 0:
            # Save progress every 20 images
            with open(captions_file, 'w') as f:
                json.dump(new_captions, f, indent=2)
    
    # Save final captions
    with open(captions_file, 'w') as f:
        json.dump(new_captions, f, indent=2)
    
    return downloaded, downloaded_images

def main():
    output_dir = Path(r'c:\Shashank\LoRA_TextToVision\datasets\gurukul_keyframes')
    api_key = 'PZh2fI3WvnlieZcM47uyspL9Xv9QHdnKjgPKDhDmaN9jJfXaxm1uzz15'
    
    # Count existing
    pexels_existing = len(list(output_dir.glob('pexels_*.png')))
    target = 200
    needed = target - pexels_existing
    
    print(f"\n{'='*70}")
    print("📸 ENHANCED PEXELS DOWNLOADER")
    print(f"{'='*70}\n")
    print(f"Current Pexels images: {pexels_existing}")
    print(f"Target: {target}")
    print(f"Need to download: {needed}\n")
    
    if needed <= 0:
        print("✅ Target already reached!")
        return
    
    downloaded, images = download_more_pexels(api_key, needed, pexels_existing, output_dir)
    
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print(f"{'='*70}\n")
    print(f"✅ Downloaded {downloaded} new Pexels images")
    print(f"📁 Total Pexels images now: {pexels_existing + downloaded}/{target}")
    
    if pexels_existing + downloaded >= target:
        print(f"\n🎉 TARGET REACHED!\n")
    else:
        print(f"\n⚠️  Still need {target - (pexels_existing + downloaded)} more\n")

if __name__ == "__main__":
    main()
