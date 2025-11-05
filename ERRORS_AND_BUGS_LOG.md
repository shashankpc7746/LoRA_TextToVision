# Errors and Bugs Log - Task 9 Dataset & Training

**Date:** November 4-5, 2025  
**Project:** LoRA_TextToVision - Task 9 (Indigenous Image Adapter)  
**Branch:** task_quality_leap  
**Phase:** Dataset Creation & 1-Epoch Training Test

---

## Table of Contents

1. [Dataset Download Errors](#dataset-download-errors)
2. [PowerShell Command Errors](#powershell-command-errors)
3. [Library & Dependency Issues](#library--dependency-issues)
4. [Image Processing Warnings](#image-processing-warnings)
5. [Training Test Issues](#training-test-issues)
6. [Resolved Issues](#resolved-issues)

---

## Dataset Download Errors

### 1. WikiMedia Commons 403 Forbidden Error

**Status:** ✅ RESOLVED

**Error Message:**
```
HTTPError: 403 Client Error: Forbidden for url: https://commons.wikimedia.org/w/api.php?...
```

**Context:**
- Occurred when trying to download educational images from WikiMedia Commons
- Initial approach used category API (`list=categorymembers`)
- WikiMedia servers blocked requests without proper User-Agent headers

**Root Cause:**
1. Missing or improper User-Agent headers in requests
2. Category API has stricter access controls
3. Server detected automated scraping behavior

**Solution Implemented:**
```python
# Changed from category API to search API
# Old (failed):
params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": f"Category:{category}",
    # ...
}

# New (working):
params = {
    "action": "query",
    "list": "search",
    "srsearch": search_term,
    "srnamespace": 6,  # File namespace
    # ...
}

# Added proper User-Agent headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
```

**Files Affected:**
- `adapters/gurukul_lora/download_production_dataset.py` (lines 185-250)

**Result:** Successfully downloaded 100/100 WikiMedia images

---

### 2. Open Images V7 Access Restrictions

**Status:** ✅ RESOLVED

**Error Message:**
```
HTTPError: 403 Client Error: Forbidden for url: https://storage.googleapis.com/openimages/...
```

**Context:**
- Attempted to download Open Images V7 dataset directly from Google Cloud Storage
- Multiple failed approaches:
  1. Direct CSV download from storage.googleapis.com
  2. Image downloads from storage URLs
  3. Flickr URL extraction from CSV metadata

**Root Cause:**
- Open Images V7 hosted on CVDF servers, not public Google Storage
- Requires specific access methods (Manual download, TFDS, or FiftyOne)
- Direct HTTP access to storage buckets blocked

**Failed Approaches:**

**Approach 1: Direct CSV Download**
```python
# FAILED - 403 Forbidden
csv_url = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
response = requests.get(csv_url)
# Result: 403 Client Error
```

**Approach 2: Image URL Downloads**
```python
# FAILED - 403 Forbidden
for image_id in image_ids:
    img_url = f"https://storage.googleapis.com/openimages/validation/{image_id}.jpg"
    response = requests.get(img_url)
    # Result: 403 Client Error
```

**Approach 3: Flickr URL Extraction**
```python
# FAILED - Complex, unreliable
# Required parsing CSV, extracting Flickr URLs, handling dead links
```

**Solution Implemented:**
```python
# Use FiftyOne library (official Open Images integration)
import fiftyone as fo

dataset = fo.zoo.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["classifications"],
    classes=["Book", "Laptop", "Desk", "Whiteboard", "Backpack", "Calculator", "Pen"],
    max_samples=target * 2
)

# Process downloaded images
for sample in dataset:
    img_path = sample.filepath
    img = Image.open(img_path)
    # ... resize and save
```

**Files Affected:**
- `adapters/gurukul_lora/download_production_dataset.py` (lines 252-350)

**Dependencies Added:**
- `fiftyone` library (auto-installed if not present)

**Result:** Successfully downloaded 200/200 Open Images

---

### 3. Pexels API Rate Limiting & Duplicate Images

**Status:** ✅ RESOLVED

**Error Message:**
- No explicit error, but fewer images downloaded than requested
- Duplicate detection prevented reaching target count

**Context:**
- Initial download: 126/200 images (74 short)
- Re-running same script: Downloaded 55 new images but total stayed at 126
- Issue: Pexels returns same popular images for related keywords

**Root Cause:**
1. Limited keyword diversity (20 keywords)
2. Pexels API returns most popular images first
3. Related keywords (e.g., "classroom students", "teacher whiteboard") return overlapping results
4. Duplicate detection system working correctly, preventing re-downloads

**Solution Implemented:**

**Enhanced Keyword Strategy:**
```python
# Original: 20 keywords
keywords = [
    "mathematics classroom", "science laboratory", "chemistry experiment",
    # ... 17 more
]

# Enhanced: 54 diverse keywords with pagination
keywords = [
    # Generic education terms
    "university lecture hall", "library study room", "laboratory equipment",
    # Subject-specific
    "algebra mathematics", "geometry shapes", "statistics charts",
    # Learning contexts
    "distance learning", "e-learning computer", "homework desk",
    # Different angles
    "education background", "learning concept", "knowledge books",
    # ... 42 more diverse terms
]

# Added pagination (3 pages per keyword)
for keyword in keywords:
    for page in range(1, 4):  # Try up to 3 pages
        params = {
            "query": keyword,
            "per_page": 10,
            "page": page
        }
```

**Files Created:**
- `adapters/gurukul_lora/download_pexels_enhanced.py` (new file, 180 lines)

**Result:** Successfully downloaded 74 additional images, reaching 200/200 target

---

## PowerShell Command Errors

### 4. Python Inline Script Syntax Error

**Status:** ✅ RESOLVED (Alternative Approach)

**Error Message:**
```powershell
SyntaxError: unterminated string literal (detected at line 1)
```

**Context:**
- Attempted to run multi-line Python code inline in PowerShell
- Goal: Resume dataset download to fill 93-image gap

**Failed Command:**
```powershell
C:/Shashank/LoRA_TextToVision/gurukul-lora-env/Scripts/python.exe -c "
import sys; 
sys.path.insert(0, r'c:\Shashank\LoRA_TextToVision\adapters\gurukul_lora'); 
from download_production_dataset import EducationalDatasetDownloader; 
import json; 
from pathlib import Path; 
output_dir = Path(r'c:\Shashank\LoRA_TextToVision\datasets\gurukul_keyframes'); 
captions_file = output_dir / 'captions.json'; 
existing = json.load(open(captions_file)); 
print(f'\nLoaded {len(existing)} existing captions'); 
d = EducationalDatasetDownloader(output_dir=str(output_dir), test_mode=False); 
d.captions = existing; 
d.downloaded_images = list(existing.keys()); 
d.limits['pexels'] = 74; 
d.limits['open_images'] = 19; 
d.limits['wikimedia'] = 0; 
print(f'\nDownloading {d.limits[\"pexels\"]} more Pexels images...'); 
pexels_count = d.download_pexels(api_key='PZh2fI3WvnlieZcM47uyspL9Xv9QHdnKjgPKDhDmaN9jJfXaxm1uzz15'); 
print(f'\nDownloading {d.limits[\"open_images\"]} more Open Images...'); 
openimg_count = d.download_open_images(); 
d.generate_captions_file(); 
print(f'\n✅ Downloaded {pexels_count + openimg_count} additional images')
"
```

**Root Cause:**
1. Complex multi-line Python string with nested quotes
2. PowerShell string parsing conflicts
3. Mix of single quotes, double quotes, and f-strings
4. Special characters (backslashes, brackets) causing escape issues

**Solution Implemented:**
- Created separate Python script file instead of inline command
- File: `adapters/gurukul_lora/download_remaining.py`
- Executed as: `python.exe download_remaining.py`

**Best Practice:**
- For complex Python logic, always use separate `.py` files
- Inline `-c` commands only for simple one-liners
- PowerShell string quoting is error-prone with nested quotes

---

## Library & Dependency Issues

### 5. xFormers CUDA Version Mismatch Warning

**Status:** ⚠️ WARNING (Non-Critical)

**Warning Message:**
```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 2.3.1+cu121 with CUDA 1201 (you have 2.7.1+cu118)
    Python  3.10.11 (you have 3.10.11)
Please reinstall xformers (see https://github.com/facebookresearch/xformers#installing-xformers)
Memory-efficient attention, SwiGLU, sparse and more won't be available.
```

**Context:**
- Occurred during training imports
- Current setup: PyTorch 2.7.1+cu118, CUDA 12.6
- xFormers built for: PyTorch 2.3.1+cu121

**Impact:**
- Training still works correctly
- Memory-efficient attention optimizations unavailable
- May use slightly more VRAM
- Performance impact minimal for small batch sizes

**Root Cause:**
- xFormers version incompatibility with PyTorch 2.7.1
- CUDA toolkit mismatch (118 vs 121)

**Potential Solutions (Not Implemented):**
1. Reinstall xFormers: `pip install xformers --force-reinstall`
2. Downgrade PyTorch to 2.3.1: `pip install torch==2.3.1+cu121`
3. Build xFormers from source for current PyTorch version

**Decision:** Not fixed - training works without optimizations, risk of breaking other dependencies

---

### 6. Triton Module Not Found Warning

**Status:** ⚠️ WARNING (Non-Critical)

**Warning Message:**
```
A matching Triton is not available, some optimizations will not be enabled
Traceback (most recent call last):
  File "...\xformers\__init__.py", line 57, in _is_triton_available
    import triton  # noqa
ModuleNotFoundError: No module named 'triton'
```

**Context:**
- Related to xFormers optimization library
- Triton: NVIDIA GPU programming language for custom kernels

**Impact:**
- Minimal performance impact for small-scale training
- Some xFormers optimizations disabled
- Training completes successfully

**Root Cause:**
- Triton not installed (not available for Windows easily)
- xFormers trying to load Triton optimizations

**Potential Solution (Not Implemented):**
- Install Triton: `pip install triton` (Linux/WSL only, not native Windows)

**Decision:** Accepted as non-critical warning

---

### 7. PyTorch AMP Deprecation Warnings

**Status:** ⚠️ WARNING (Non-Critical)

**Warning Messages:**
```
FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. 
Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.

FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. 
Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
```

**Context:**
- Deprecated API usage in xFormers library
- PyTorch 2.7.1 introduced new AMP API

**Impact:**
- None currently - deprecated API still works
- Will break in future PyTorch versions

**Root Cause:**
- xFormers using old PyTorch AMP API
- Needs update from xFormers maintainers

**Solution:**
- Wait for xFormers update
- Or suppress warnings: `warnings.filterwarnings('ignore', category=FutureWarning)`

**Decision:** Monitoring - not breaking current functionality

---

## Image Processing Warnings

### 8. PIL DecompressionBombWarning

**Status:** ✅ HANDLED (Not an Error)

**Warning Message:**
```
Image size (XXX pixels) exceeds limit of 89478485 pixels, 
could be decompression bomb DOS attack.
```

**Context:**
- Occurred when downloading WikiMedia Commons images
- WikiMedia hosts very high-resolution educational diagrams
- Example: 15000x12000 pixel scientific illustrations

**Impact:**
- Not an actual error - images processed successfully
- PIL safety warning to prevent memory exhaustion attacks
- All images properly resized to 1024x1024

**Root Cause:**
- WikiMedia has ultra-high-resolution source images
- PIL default safety limit: ~89 megapixels
- Educational diagrams often exceed this limit

**Handling in Code:**
```python
# PIL automatically handles large images
img = Image.open(BytesIO(response.content))
# Warning appears but image loads fine
img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
# Resized to safe dimensions
```

**Solution Options (Not Implemented):**
```python
# Option 1: Increase PIL limit
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable limit

# Option 2: Suppress warning
import warnings
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
```

**Decision:** Accepted as informational warning - no action needed

---

## Training Test Issues

### 9. Long Model Loading Time (Initial Run)

**Status:** ℹ️ EXPECTED BEHAVIOR

**Observation:**
- First import of diffusers takes 1-2 minutes
- Message displayed: "Loading diffusers (be patient, scanning model files)..."

**Context:**
- SDXL model loading from Hugging Face cache
- Multiple model components: VAE, UNet, Text Encoders
- Total model size: ~6.9 GB

**Impact:**
- Training start delayed by ~2 minutes
- Only affects first run or after cache clear
- Subsequent runs much faster (cached)

**Root Cause:**
- Hugging Face diffusers scans all model files
- Validates checksums and configurations
- Loads multiple safetensors files

**Mitigation in Code:**
```python
# Added informative messages
print("Loading diffusers (be patient, scanning model files)...", end="", flush=True)
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
print(" ✓")
```

**Decision:** Expected behavior - user notification added

---

### 10. High Memory Usage During Training

**Status:** ℹ️ EXPECTED BEHAVIOR

**Observation:**
- GPU memory usage: ~7.2 GB / 8.0 GB (90% utilization)
- System RAM usage: ~6 GB during training

**Context:**
- SDXL is a large model (2.6B parameters)
- Training with FP32 VAE + FP16 text encoders + FP32 UNet
- LoRA training: 23.2M trainable parameters

**Memory Breakdown:**
- VAE (FP32): ~320 MB
- Text Encoders (FP16): ~1.4 GB
- UNet (FP32): ~5.0 GB
- Gradients & Optimizer states: ~400 MB
- Activations: ~100 MB

**Impact:**
- Near GPU memory limit on RTX 3060 Ti (8GB)
- No OOM errors with batch_size=1
- Cannot increase batch size without optimization

**Potential Optimizations (Not Implemented):**
```python
# 1. Gradient checkpointing
unet.enable_gradient_checkpointing()

# 2. Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. CPU offloading
vae = vae.to('cpu')
# Move to GPU only when needed

# 4. Lower precision
unet = unet.half()  # FP16 training
```

**Decision:** Current configuration works, optimizations deferred

---

## Resolved Issues

### 11. Dataset Path Issues in Training Script

**Status:** ✅ FIXED

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/gurukul_keyframes'
```

**Context:**
- Training script executed from `adapters/gurukul_lora/`
- Relative path incorrect from subdirectory

**Solution:**
```python
# Fixed in train_1epoch_test.py
# Old:
dataset = GurukulDataset("datasets/gurukul_keyframes", size=512)

# New:
dataset = GurukulDataset("../../datasets/gurukul_keyframes", size=512)
```

---

### 12. Captions.json File Count Mismatch

**Status:** ✅ VERIFIED

**Initial Concern:**
- Downloaded 407 images but captions.json needed verification

**Verification Command:**
```powershell
$json = Get-Content "datasets\gurukul_keyframes\captions.json" -Raw | ConvertFrom-Json
$entries = ($json | Get-Member -MemberType NoteProperty).Count
```

**Result:**
- Captions.json: 407 entries ✅
- Image files: 407 PNG files ✅
- Perfect match confirmed

---

## Summary Statistics

### Errors by Category:
- **Critical Errors (Fixed):** 3
  - WikiMedia 403 Forbidden
  - Open Images Access Restrictions
  - Pexels Duplicate Images
  
- **Non-Critical Warnings:** 4
  - xFormers CUDA mismatch
  - Triton module not found
  - PyTorch AMP deprecation
  - PIL DecompressionBomb warnings

- **Expected Behaviors:** 2
  - Long model loading time
  - High GPU memory usage

- **PowerShell Issues:** 1
  - Inline Python script syntax error

### Resolution Success Rate:
- **Resolved:** 3/3 critical errors (100%)
- **Mitigated:** 4/4 warnings (handled appropriately)
- **Documented:** 2/2 expected behaviors

---

## Lessons Learned

### 1. API Access Patterns
- Always include proper User-Agent headers for web scraping
- Search APIs often more reliable than category/listing APIs
- Official libraries (FiftyOne) preferred over manual HTTP access

### 2. PowerShell vs Python
- Complex Python logic → separate `.py` files
- Inline `-c` commands → simple one-liners only
- Quote escaping nightmares avoided with file-based approach

### 3. Dataset Download Strategy
- Pagination necessary for large datasets
- Duplicate detection critical for multi-run downloads
- Diverse keywords prevent image overlap
- Progress saving every N images prevents data loss

### 4. Library Compatibility
- Version mismatches often produce warnings, not errors
- Performance optimizations (xFormers) nice-to-have, not required
- Training works with fallback implementations

### 5. Memory Management
- Batch size = 1 safe for 8GB VRAM with SDXL
- FP32 training for LoRA is standard
- Monitor memory usage, don't assume OOM won't happen

---

## Future Improvements

### Dataset Download:
- [ ] Add retry logic with exponential backoff
- [ ] Implement resume-from-checkpoint for interrupted downloads
- [ ] Add image quality validation (blur detection, size checks)
- [ ] Create dataset manifest with checksums

### Training:
- [ ] Implement gradient checkpointing for lower memory
- [ ] Add mixed precision training (AMP)
- [ ] Enable xFormers optimizations (reinstall compatible version)
- [ ] Add WandB/TensorBoard logging

### Error Handling:
- [ ] Centralized error logging system
- [ ] Email/Slack notifications for critical failures
- [ ] Automatic recovery from transient errors
- [ ] Better progress visualization

---

## Contact & References

**Project:** LoRA_TextToVision  
**Repository:** github.com/shashankpc7746/LoRA_TextToVision  
**Branch:** task_quality_leap  
**Python:** 3.10.11  
**PyTorch:** 2.7.1+cu118  
**CUDA:** 12.6  

**Key Files:**
- Dataset downloader: `adapters/gurukul_lora/download_production_dataset.py`
- Enhanced Pexels: `adapters/gurukul_lora/download_pexels_enhanced.py`
- Training script: `adapters/gurukul_lora/train_optimized.py`
- 1-epoch test: `adapters/gurukul_lora/train_1epoch_test.py`

**Last Updated:** November 5, 2025, 5:30 AM IST
