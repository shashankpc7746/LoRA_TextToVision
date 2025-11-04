"""
Production Dataset Downloader for Gurukul LoRA Training
Downloads curated educational images from 3 sources:
1. Open Images V7 (Google Research)
2. Pexels API (Professional Stock Photos)
3. WikiMedia Commons (Educational Diagrams)

Total: 500 images (200 + 200 + 100)
"""

import os
import sys
import json
import requests
import time
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse


class EducationalDatasetDownloader:
    """Download curated educational images from multiple sources"""
    
    def __init__(self, output_dir="../../datasets/gurukul_keyframes", test_mode=False):
        self.output_dir = Path(output_dir).resolve()  # Convert to absolute path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_mode = test_mode
        
        # For test mode, download fewer images
        self.limits = {
            'pexels': 5 if test_mode else 200,
            'wikimedia': 3 if test_mode else 100,
            'open_images': 2 if test_mode else 200
        }
        
        self.downloaded_images = []
        self.captions = {}
        
    def download_image(self, url, filename, caption="", custom_headers=None):
        """Download and save image from URL"""
        try:
            # Default headers to avoid 403 errors
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            if custom_headers:
                headers.update(custom_headers)
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Load image
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 1024x1024 (training resolution)
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Save
            filepath = self.output_dir / filename
            img.save(filepath, 'PNG', quality=95)
            
            # Store metadata
            self.downloaded_images.append(filename)
            self.captions[filename] = caption
            
            return True
            
        except Exception as e:
            # Suppress verbose error messages in quiet mode
            if not hasattr(self, '_quiet_errors'):
                print(f"  ⚠️  Failed to download {filename}: {str(e)[:80]}...")
            return False
    
    def download_pexels(self, api_key=None):
        """Download from Pexels API (Professional Stock Photos)"""
        print(f"\n{'='*70}")
        print("📸 PEXELS - Professional Stock Photos")
        print(f"{'='*70}\n")
        
        # Check for API key
        if not api_key:
            api_key = os.environ.get('PEXELS_API_KEY')
        
        if not api_key:
            print("⚠️  PEXELS_API_KEY not found in environment variables")
            print("\n📋 To get a free API key:")
            print("   1. Visit: https://www.pexels.com/api/")
            print("   2. Sign up (free, no credit card)")
            print("   3. Copy your API key")
            print("   4. Set environment variable:")
            print("      Windows: $env:PEXELS_API_KEY='your_key_here'")
            print("      Linux/Mac: export PEXELS_API_KEY='your_key_here'")
            print("\n⏭️  Skipping Pexels download...\n")
            return 0
        
        # Educational keywords
        keywords = [
            "mathematics classroom", "science laboratory", "chemistry experiment",
            "biology classroom", "physics experiment", "computer programming",
            "history books", "geography map", "literature library",
            "music lesson", "art class painting", "musical instruments",
            "business presentation", "engineering workspace", "medical education",
            "classroom students", "teacher whiteboard", "online learning",
            "study desk books", "educational technology"
        ]
        
        api_url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        
        target = self.limits['pexels']
        per_keyword = max(1, target // len(keywords))
        downloaded = 0
        
        print(f"Target: {target} images ({per_keyword} per keyword)")
        print(f"Keywords: {len(keywords)}\n")
        
        for keyword in tqdm(keywords, desc="Downloading from Pexels"):
            if downloaded >= target:
                break
            
            try:
                params = {
                    "query": keyword,
                    "per_page": per_keyword,
                    "orientation": "square",
                    "size": "large"
                }
                
                response = requests.get(api_url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                
                photos = response.json().get('photos', [])
                
                for photo in photos:
                    if downloaded >= target:
                        break
                    
                    url = photo['src']['original']
                    filename = f"pexels_{photo['id']:08d}.png"
                    caption = f"{keyword} - professional educational photo"
                    
                    if self.download_image(url, filename, caption):
                        downloaded += 1
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"\n  ⚠️  Error with keyword '{keyword}': {e}")
                continue
        
        print(f"\n✅ Downloaded {downloaded} images from Pexels")
        return downloaded
    
    def download_wikimedia(self):
        """Download from WikiMedia Commons (Educational Diagrams)"""
        print(f"\n{'='*70}")
        print("📚 WIKIMEDIA COMMONS - Educational Diagrams")
        print(f"{'='*70}\n")
        
        # Educational search terms
        search_terms = [
            "mathematics diagram", "chemistry diagram", "physics diagram",
            "biology diagram", "geometry diagram", "scientific illustration",
            "educational chart", "learning diagram", "school science",
            "textbook illustration"
        ]
        
        api_url = "https://commons.wikimedia.org/w/api.php"
        target = self.limits['wikimedia']
        per_term = max(1, target // len(search_terms)) + 1
        downloaded = 0
        
        # Browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        print(f"Target: {target} images ({per_term} per search term)")
        print(f"Search terms: {len(search_terms)}\n")
        
        self._quiet_errors = True  # Suppress verbose errors
        
        for term in tqdm(search_terms, desc="Downloading from WikiMedia"):
            if downloaded >= target:
                break
            
            try:
                # Search for images
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srnamespace": "6",  # File namespace
                    "srlimit": per_term * 3,  # Get more to filter
                    "format": "json"
                }
                
                response = requests.get(api_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                results = response.json().get('query', {}).get('search', [])
                
                for result in results:
                    if downloaded >= target:
                        break
                    
                    try:
                        # Get image URL
                        title = result['title']
                        img_params = {
                            "action": "query",
                            "titles": title,
                            "prop": "imageinfo",
                            "iiprop": "url|size",
                            "format": "json"
                        }
                        
                        img_response = requests.get(api_url, params=img_params, headers=headers, timeout=10)
                        pages = img_response.json().get('query', {}).get('pages', {})
                        
                        for page in pages.values():
                            if 'imageinfo' in page and page['imageinfo']:
                                url = page['imageinfo'][0].get('url', '')
                                width = page['imageinfo'][0].get('width', 0)
                                height = page['imageinfo'][0].get('height', 0)
                                
                                # Only download reasonably sized images (not SVG, PDF)
                                if (url.lower().endswith(('.jpg', '.jpeg', '.png')) and 
                                    width >= 512 and height >= 512):
                                    
                                    filename = f"wiki_{result['pageid']:08d}.png"
                                    caption = f"{term} - educational diagram from WikiMedia"
                                    
                                    if self.download_image(url, filename, caption):
                                        downloaded += 1
                                        break
                        
                        time.sleep(0.3)  # Rate limiting between images
                        
                    except Exception:
                        continue
                
                time.sleep(0.5)  # Rate limiting between searches
                
            except Exception as e:
                print(f"\n  ⚠️  Search '{term}' failed: {str(e)[:60]}...")
                continue
        
        delattr(self, '_quiet_errors')
        print(f"\n✅ Downloaded {downloaded} images from WikiMedia")
        return downloaded
    
    def download_open_images(self):
        """Download from Open Images V7 using FiftyOne library"""
        print(f"\n{'='*70}")
        print("🔍 OPEN IMAGES V7 - Using FiftyOne (Official Integration)")
        print(f"{'='*70}\n")
        
        print("📝 Using FiftyOne library for easy Open Images access")
        print("   Downloads directly from CVDF servers\n")
        
        try:
            # Try to install fiftyone if not present
            try:
                import fiftyone as fo
                print("✅ FiftyOne already installed\n")
            except ImportError:
                print("📦 Installing FiftyOne library (this may take a minute)...")
                import subprocess
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "fiftyone"],
                    capture_output=True, text=True, timeout=180
                )
                if result.returncode != 0:
                    print("⚠️  Could not install FiftyOne")
                    print("💡 Install manually: pip install fiftyone")
                    print("⏭️  Skipping Open Images (Pexels + WikiMedia still work!)\n")
                    return 0
                import fiftyone as fo
                print("✅ FiftyOne installed successfully\n")
            
            # Educational classes for Open Images V7
            educational_classes = [
                "Book", "Laptop", "Desk", "Whiteboard",
                "Backpack", "Calculator", "Pen", "Paper"
            ]
            
            target = self.limits['open_images']
            per_class = max(1, target // len(educational_classes))
            
            print(f"Target: {target} images ({per_class} per class)")
            print(f"Classes: {', '.join(educational_classes)}")
            print("📥 Downloading from Open Images V7 via FiftyOne...")
            print("   (First-time setup may take a few minutes)\n")
            
            # Download Open Images V7 subset using FiftyOne
            downloaded = 0
            temp_dataset_name = f"oi_temp_{int(time.time())}"
            
            try:
                # Load Open Images validation set with specific classes
                dataset = fo.zoo.load_zoo_dataset(
                    "open-images-v7",
                    split="validation",
                    label_types=["classifications"],
                    classes=educational_classes,
                    max_samples=target * 2,  # Get extra to filter
                    dataset_name=temp_dataset_name
                )
                
                print(f"✅ Loaded {len(dataset)} images from Open Images\n")
                print("� Converting and saving images...\n")
                
                self._quiet_errors = True
                
                for idx, sample in enumerate(tqdm(dataset, desc="Processing Open Images", total=min(target, len(dataset)))):
                    if downloaded >= target:
                        break
                    
                    try:
                        # Get image path from FiftyOne
                        img_path = sample.filepath
                        
                        # Get classifications
                        classes = []
                        if sample.positive_labels and sample.positive_labels.classifications:
                            classes = [c.label for c in sample.positive_labels.classifications[:2]]
                        
                        if not classes:
                            continue
                        
                        # Load and process image
                        img = Image.open(img_path)
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                        
                        # Save
                        classes_str = "_".join(classes)
                        filename = f"openimg_{classes_str}_{sample.id[:8]}.png"
                        filepath = self.output_dir / filename
                        img.save(filepath, 'PNG', quality=95)
                        
                        # Store metadata
                        self.downloaded_images.append(filename)
                        caption = f"{', '.join(classes)} - from Open Images V7"
                        self.captions[filename] = caption
                        downloaded += 1
                        
                    except Exception as e:
                        continue
                
                # Cleanup FiftyOne dataset
                fo.delete_dataset(temp_dataset_name)
                
                if hasattr(self, '_quiet_errors'):
                    delattr(self, '_quiet_errors')
                
                print(f"\n✅ Downloaded {downloaded} images from Open Images V7")
                return downloaded
                
            except Exception as e:
                print(f"⚠️  FiftyOne download failed: {e}")
                # Try to cleanup
                try:
                    fo.delete_dataset(temp_dataset_name)
                except:
                    pass
                print("⏭️  Continuing with other sources...\n")
                return 0
            
        except Exception as e:
            print(f"❌ Open Images download failed: {e}")
            print("⏭️  Continuing with other sources...\n")
            return 0
    
    def generate_captions_file(self):
        """Generate captions.json file"""
        caption_file = self.output_dir / "captions.json"
        with open(caption_file, 'w') as f:
            json.dump(self.captions, f, indent=2)
        print(f"\n✅ Generated {caption_file}")
    
    def validate_dataset(self):
        """Validate all downloaded images"""
        print(f"\n{'='*70}")
        print("✓ VALIDATION")
        print(f"{'='*70}\n")
        
        valid = 0
        invalid = []
        
        for filename in tqdm(self.downloaded_images, desc="Validating images"):
            try:
                img = Image.open(self.output_dir / filename)
                if img.size == (1024, 1024) and img.mode == 'RGB':
                    valid += 1
                else:
                    invalid.append(f"{filename} - Wrong size or mode")
                img.close()
            except Exception as e:
                invalid.append(f"{filename} - {e}")
        
        print(f"\n✅ Valid images: {valid}")
        if invalid:
            print(f"⚠️  Invalid images: {len(invalid)}")
            for inv in invalid[:5]:  # Show first 5
                print(f"   - {inv}")
        
        return valid, invalid
    
    def run(self, pexels_api_key=None):
        """Execute complete download pipeline"""
        print(f"\n{'='*70}")
        print("🚀 EDUCATIONAL DATASET DOWNLOADER")
        print(f"{'='*70}")
        
        mode = "TEST MODE (10 images)" if self.test_mode else "PRODUCTION MODE (500 images)"
        print(f"\nMode: {mode}")
        print(f"Output: {self.output_dir}")
        print(f"\nTargets:")
        print(f"  - Pexels: {self.limits['pexels']} images")
        print(f"  - WikiMedia: {self.limits['wikimedia']} images")
        print(f"  - Open Images: {self.limits['open_images']} images")
        print(f"  - Total: {sum(self.limits.values())} images\n")
        
        print("🚀 Starting download...\n")
        
        # Download from all sources
        pexels_count = self.download_pexels(api_key=pexels_api_key)
        wikimedia_count = self.download_wikimedia()
        openimages_count = self.download_open_images()
        
        total = pexels_count + wikimedia_count + openimages_count
        
        # Generate captions
        if total > 0:
            self.generate_captions_file()
            
            # Validate
            valid, invalid = self.validate_dataset()
            
            # Summary
            print(f"\n{'='*70}")
            print("📊 DOWNLOAD COMPLETE")
            print(f"{'='*70}\n")
            print(f"Sources:")
            print(f"  - Pexels: {pexels_count} images")
            print(f"  - WikiMedia: {wikimedia_count} images")
            print(f"  - Open Images: {openimages_count} images")
            print(f"\nTotal Downloaded: {total} images")
            print(f"Valid: {valid} images")
            print(f"Location: {self.output_dir}")
            print(f"\n✅ Dataset ready for training!\n")
        else:
            print("\n❌ No images downloaded. Check API keys and connections.\n")


def main():
    parser = argparse.ArgumentParser(description="Download educational dataset for Gurukul LoRA")
    parser.add_argument("--test", action="store_true", 
                       help="Test mode: Download only 10 images")
    # Use absolute path by default (from script location)
    default_output = Path(__file__).parent.parent.parent / "datasets" / "gurukul_keyframes"
    parser.add_argument("--output", type=str, default=str(default_output),
                       help="Output directory for dataset")
    parser.add_argument("--pexels-key", type=str, default=None,
                       help="Pexels API key (or use PEXELS_API_KEY env variable)")
    
    args = parser.parse_args()
    
    downloader = EducationalDatasetDownloader(
        output_dir=args.output,
        test_mode=args.test
    )
    
    downloader.run(pexels_api_key=args.pexels_key)


if __name__ == "__main__":
    main()
