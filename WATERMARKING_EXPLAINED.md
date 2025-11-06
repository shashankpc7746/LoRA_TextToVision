# Watermarking Implementation - Detailed Explanation

## Question: How Does Watermarking Work?

You asked about `.watermark.json` files and whether we're adding watermark images to videos. Great question! Let me explain the **three different watermarking approaches** we've implemented:

---

## 🎯 Three Watermarking Methods

### Method 1: FFmpeg Metadata Watermarking (RECOMMENDED) ⭐

**How it works:**
- Embeds watermark data **inside the video file's metadata tags** (like MP4 metadata)
- Uses `ffmpeg` to add custom tags: `BHIV_WATERMARK` and `BUILD_ID`
- **NO visible watermark image** on the video
- **NO separate file needed** - watermark is embedded IN the video
- Fast (just metadata copy, no re-encoding)

**Example:**
```python
from security import embed_watermark

# Embeds metadata inside the MP4 file
watermarked_video = embed_watermark("lesson_video.mp4", build_id="build_20251106_001")
# Output: lesson_video_watermarked.mp4 (with embedded metadata)
```

**FFmpeg command used internally:**
```bash
ffmpeg -i input.mp4 \
  -metadata BHIV_WATERMARK="base64_encoded_data" \
  -metadata BUILD_ID="build_20251106_001" \
  -c copy \  # Copy codec (no re-encoding)
  output.mp4
```

**Detection:**
```python
from security import detect_watermark

result = detect_watermark("lesson_video_watermarked.mp4")
print(result)
# {
#   'found': True,
#   'build_id': 'build_20251106_001',
#   'metadata': {...},
#   'detection_method': 'ffmpeg_metadata'
# }
```

**Advantages:**
- ✅ No visual impact on video
- ✅ Fast (no re-encoding)
- ✅ Embedded in file (can't be separated)
- ✅ Works with all video players
- ✅ Production-ready

**Limitations:**
- ⚠️ Can be stripped if someone re-encodes the video
- ⚠️ Requires ffmpeg installed

---

### Method 2: LSB Watermarking (ADVANCED) 🔬

**How it works:**
- Embeds watermark in the **Least Significant Bits** of video frames
- Modifies pixel values imperceptibly (e.g., RGB 255 → 254)
- **Invisible to human eye** but detectable algorithmically
- Survives compression to some degree

**Current Status:** 
- ⚠️ **Placeholder implementation** (TODO in code)
- Would require opencv-python or moviepy to manipulate frames
- More robust than metadata but slower

**How it WOULD work (when fully implemented):**
```python
# Example of LSB embedding (conceptual)
for frame in video_frames:
    for i, bit in enumerate(watermark_pattern):
        # Embed bit in LSB of pixel
        frame[0, 0, i % 3] = (frame[0, 0, i % 3] & 0xFE) | bit
```

**Current fallback:**
- Creates `.watermark.json` sidecar file (temporary until LSB is implemented)
- This is just a development workaround

**Advantages (when implemented):**
- ✅ Survives re-encoding (to some degree)
- ✅ Invisible watermark
- ✅ More robust than metadata

**Limitations:**
- ⚠️ Slower (needs to process every frame)
- ⚠️ Complex implementation
- ⚠️ Not yet fully implemented

---

### Method 3: Sidecar File Fallback (.watermark.json) 📄

**How it works:**
- Creates a **separate JSON file** alongside the video
- Named: `video.mp4.watermark.json`
- Contains watermark metadata in plain JSON
- **Only used as fallback** when ffmpeg/LSB unavailable

**When it's created:**
1. If `ffmpeg` is not installed
2. During development/testing
3. As backup verification method

**Example `.watermark.json` file:**
```json
{
  "build_id": "build_20251106_001",
  "watermark_pattern": [0, 1, 1, 0, 1, 0, ...],  // 32-bit pattern
  "watermarked_at": "2025-11-06T10:30:00Z",
  "original_file": "lesson_video.mp4"
}
```

**Advantages:**
- ✅ Always works (no dependencies)
- ✅ Easy to read and verify
- ✅ Good for development/testing

**Limitations:**
- ❌ Can be deleted separately from video
- ❌ Not production-grade
- ❌ Intended as fallback only

---

## 🎨 NO Visible Watermark Image!

**Important:** We are **NOT** adding visible watermark images like this:

```
┌─────────────────────┐
│                     │
│   Video Content     │
│                     │
│         ┌──────────┐│ ← NO "BHIV" logo overlay
│         │ BHIV ©   ││ ← NO text watermark
│         └──────────┘│ ← NO visible branding
└─────────────────────┘
```

Instead, our watermarks are:
1. **Metadata tags** (invisible, embedded in file)
2. **LSB pixel modifications** (invisible, < 1 bit per pixel change)
3. **Sidecar JSON** (separate file, fallback only)

---

## 🔍 How Detection Works

### Step 1: Try FFmpeg Metadata
```python
# Use ffprobe to read metadata tags
ffprobe -v quiet -print_format json -show_format video.mp4

# Output includes:
{
  "format": {
    "tags": {
      "BHIV_WATERMARK": "eyJidWlsZF9pZCI6ImJ1aWxkXzIwMjUxMTA2XzAwMSJ9",
      "BUILD_ID": "build_20251106_001"
    }
  }
}
```

### Step 2: Check for Sidecar File
```python
# If no metadata found, check for .watermark.json
if os.path.exists("video.mp4.watermark.json"):
    with open("video.mp4.watermark.json") as f:
        watermark_data = json.load(f)
```

### Step 3: Try LSB Extraction (Future)
```python
# When implemented: extract bits from frame pixels
pattern = extract_lsb_pattern(video_frames)
build_id = decode_pattern(pattern)
```

---

## 📦 Content Fingerprinting (Bonus)

We also compute **cryptographic hashes** for video files:

```python
from security import compute_fingerprint

fingerprint = compute_fingerprint("video.mp4", build_id="build_20251106_001")

print(fingerprint)
# {
#   'filename': 'video.mp4',
#   'build_id': 'build_20251106_001',
#   'sha256': 'a1b2c3d4e5f6...',  # File content hash
#   'blake2b': 'x1y2z3a4b5c6...',  # Faster hash
#   'file_size': 15234567,
#   'created_at': '2025-11-06T10:30:00Z',
#   'perceptual_hash': 'f9e8d7c6b5a4'  # Survives compression
# }
```

**Use cases:**
- Detect exact copies (SHA256 match)
- Detect re-encoded copies (perceptual hash similar)
- Build registry of authorized videos

---

## 🚀 Production Recommendation

**For Production Use:**

1. **Primary Method:** FFmpeg Metadata Watermarking
   - Fast, reliable, no visual impact
   - Set `BUILD_ID` environment variable
   - Embed metadata in all generated videos

2. **Backup Method:** Content Fingerprinting
   - Compute SHA256 + perceptual hash
   - Store in BHIV registry
   - Use for detection/comparison

3. **Future Enhancement:** LSB Watermarking
   - Implement when opencv-python available
   - More robust against re-encoding
   - Combine with metadata for double protection

**DO NOT use sidecar files (.watermark.json) in production** - they're just for development/testing!

---

## 📝 Example: Complete Video Generation Flow

```python
import os
from security import embed_watermark, compute_fingerprint
from audit_logger import get_audit_logger

def generate_secure_video(prompt: str, ksml_token: str):
    # 1. Generate video
    video_path = generate_video(prompt)  # Your existing function
    
    # 2. Set BUILD_ID (from CI or manual)
    build_id = os.getenv('BUILD_ID', f'build_{datetime.now().strftime("%Y%m%d")}_001')
    
    # 3. Embed watermark (uses ffmpeg metadata by default)
    watermarked_path = embed_watermark(video_path, build_id=build_id)
    # Result: video_watermarked.mp4 (with embedded metadata tags)
    
    # 4. Compute fingerprint
    fingerprint = compute_fingerprint(watermarked_path, build_id=build_id)
    
    # 5. Log to audit trail
    audit_logger = get_audit_logger()
    audit_logger.log_video_generation(
        prompt=prompt,
        output_path=watermarked_path,
        ksml_token={"ksml_token": ksml_token},
        security_metadata={
            "build_id": build_id,
            "artifact_hash": fingerprint['sha256'],
            "watermark_method": "ffmpeg_metadata",
            "watermark_id": build_id
        }
    )
    
    return watermarked_path

# Usage
video = generate_secure_video(
    prompt="Ancient Indian classroom scene",
    ksml_token="ksml_abc123"
)

print(f"✅ Secure video generated: {video}")
print(f"✅ Watermark embedded (invisible, in metadata)")
print(f"✅ Can be verified with: python tools/detect_provenance.py {video}")
```

---

## 🔍 Verification Example

```bash
# Check if video has watermark
python tools/detect_provenance.py lesson_video_watermarked.mp4

# Output:
# ======================================================================
# PROVENANCE REPORT
# ======================================================================
# 
# File: lesson_video_watermarked.mp4
# Size: 15,234,567 bytes
# Type: .mp4
# 
# ======================================================================
# PROVENANCE STATUS
# ======================================================================
# ✅ VERIFIED - File has valid provenance
#    Build ID: build_20251106_001
# 
# ======================================================================
# WATERMARK
# ======================================================================
# ✅ Watermark detected
#    Build ID: build_20251106_001
#    Method: ffmpeg_metadata  ← Embedded in video file
# 
# ======================================================================
# CONTENT FINGERPRINT
# ======================================================================
# SHA256:  a1b2c3d4e5f6...
# BLAKE2b: x1y2z3a4b5c6...
```

---

## Summary

| Method | Visibility | File Type | Robustness | Status | Production |
|--------|-----------|-----------|------------|--------|-----------|
| **FFmpeg Metadata** | Invisible | Embedded | Medium | ✅ Complete | ✅ Recommended |
| **LSB Watermarking** | Invisible | Embedded | High | ⏳ TODO | 🔮 Future |
| **Sidecar JSON** | N/A | Separate | Low | ✅ Fallback | ❌ Dev Only |

**Bottom Line:**
- ✅ We use **invisible watermarks** (no visible "BHIV" logo)
- ✅ Primary method: **FFmpeg metadata tags** (embedded in video)
- ✅ Backup: **Content fingerprinting** (SHA256 hashes)
- ⏳ Future: **LSB watermarking** (pixel-level embedding)
- ❌ `.watermark.json` is just a **development fallback**, not used in production

The watermark is **completely invisible** to viewers but **detectable by our tools**!
