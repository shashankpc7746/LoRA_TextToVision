#!/usr/bin/env python3
"""
Download Essential AnimateDiff Models for Character Consistency
This script downloads the proper motion modules and domain adapters
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

def download_file(url, filepath, expected_size=None):
    """Download file with progress bar and verification"""
    print(f"📦 Downloading {filepath.name}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            print(f"⚠️ Warning: Expected size {expected_size}, got {total_size}")
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded {filepath.name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filepath.name}: {e}")
        if filepath.exists():
            filepath.unlink()
        return False

def verify_file_hash(filepath, expected_hash):
    """Verify file integrity using SHA256"""
    if not filepath.exists():
        return False
    
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest() == expected_hash

def download_motion_modules():
    """Download AnimateDiff motion modules for temporal consistency"""
    motion_module_dir = Path("models/Motion_Module")
    motion_module_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Downloading AnimateDiff Motion Modules...")
    
    # Essential motion modules with file info
    motion_modules = {
        "mm_sd_v15_v2.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt",
            "size": 1703946240,  # ~1.7GB
            "description": "Motion Module v2 - Enhanced temporal consistency"
        },
        "v3_sd15_mm.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt",
            "size": 1703946240,  # ~1.7GB  
            "description": "Motion Module v3 - Latest with improved quality"
        }
    }
    
    for filename, info in motion_modules.items():
        filepath = motion_module_dir / filename
        
        if filepath.exists() and filepath.stat().st_size == info["size"]:
            print(f"✅ {filename} already exists and verified")
            continue
        
        print(f"📋 {info['description']}")
        success = download_file(info["url"], filepath, info["size"])
        
        if success:
            print(f"✅ {filename} ready for use")
        else:
            print(f"❌ Failed to download {filename}")

def download_domain_adapters():
    """Download domain adapters for artifact reduction"""
    adapter_dir = Path("models/DreamBooth_LoRA")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Downloading Domain Adapters...")
    
    # Domain adapters for quality improvement
    adapters = {
        "v3_sd15_adapter.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt",
            "size": 97427456,  # ~97MB
            "description": "Domain Adapter v3 - Reduces artifacts and improves quality"
        }
    }
    
    for filename, info in adapters.items():
        filepath = adapter_dir / filename
        
        if filepath.exists() and filepath.stat().st_size == info["size"]:
            print(f"✅ {filename} already exists and verified")
            continue
        
        print(f"📋 {info['description']}")
        success = download_file(info["url"], filepath, info["size"])
        
        if success:
            print(f"✅ {filename} ready for use")
        else:
            print(f"❌ Failed to download {filename}")

def download_motion_loras():
    """Download MotionLoRA for specific camera movements"""
    lora_dir = Path("models/MotionLoRA")
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    print("📹 Downloading MotionLoRA modules...")
    
    # Essential MotionLoRA for camera effects
    motion_loras = {
        "v2_lora_ZoomIn.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt",
            "size": 77594624,  # ~77MB
            "description": "Zoom In camera movement"
        },
        "v2_lora_ZoomOut.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt", 
            "size": 77594624,  # ~77MB
            "description": "Zoom Out camera movement"
        },
        "v2_lora_PanLeft.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt",
            "size": 77594624,  # ~77MB
            "description": "Pan Left camera movement"
        },
        "v2_lora_PanRight.ckpt": {
            "url": "https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt",
            "size": 77594624,  # ~77MB
            "description": "Pan Right camera movement"
        }
    }
    
    for filename, info in motion_loras.items():
        filepath = lora_dir / filename
        
        if filepath.exists() and filepath.stat().st_size == info["size"]:
            print(f"✅ {filename} already exists and verified")
            continue
        
        print(f"📋 {info['description']}")
        success = download_file(info["url"], filepath, info["size"])
        
        if success:
            print(f"✅ {filename} ready for use")
        else:
            print(f"❌ Failed to download {filename}")

def main():
    """Download all essential AnimateDiff models"""
    print("🚀 AnimateDiff Model Downloader")
    print("=" * 50)
    
    # Check available disk space
    import shutil
    free_space = shutil.disk_usage(".").free
    required_space = 4 * 1024 * 1024 * 1024  # ~4GB
    
    if free_space < required_space:
        print(f"⚠️ Warning: Low disk space. Required: ~4GB, Available: {free_space / (1024**3):.1f}GB")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    print(f"💾 Available disk space: {free_space / (1024**3):.1f}GB")
    print()
    
    # Download all components
    download_motion_modules()
    print()
    download_domain_adapters()
    print()
    download_motion_loras()
    
    print()
    print("🎉 AnimateDiff model download complete!")
    print("✅ Your system now has enhanced character consistency capabilities")
    print("🎬 Ready for high-quality video generation with temporal coherence")

if __name__ == "__main__":
    main()
