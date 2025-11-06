"""
Watermarking Demo Script
Shows how watermarking actually works in practice
"""
from security import VideoWatermarker, embed_watermark, detect_watermark
import tempfile
import os

print("\n" + "="*70)
print("WATERMARKING DEMONSTRATION")
print("="*70)

# Demo 1: Watermark Pattern Generation
print("\n1️⃣  DETERMINISTIC WATERMARK PATTERN\n")
print("   Watermark is generated from BUILD_ID (like a fingerprint)")

wm = VideoWatermarker('demo_build_001')
pattern1 = wm.generate_watermark_pattern('demo_build_001')
print(f"   BUILD_ID: demo_build_001")
print(f"   Pattern (first 16 bits): {pattern1[:16]}")
print(f"   Binary: {''.join(map(str, pattern1[:16]))}")

print("\n   Same BUILD_ID always produces same pattern:")
pattern2 = wm.generate_watermark_pattern('demo_build_001')
print(f"   Pattern again: {pattern2[:16]}")
print(f"   ✅ Patterns match: {(pattern1 == pattern2).all()}")

print("\n   Different BUILD_ID produces different pattern:")
pattern3 = wm.generate_watermark_pattern('demo_build_002')
print(f"   BUILD_ID: demo_build_002")
print(f"   Pattern: {pattern3[:16]}")
print(f"   ✅ Different from first: {not (pattern1 == pattern3).all()}")

# Demo 2: Watermark Embedding
print("\n" + "="*70)
print("2️⃣  WATERMARK EMBEDDING (3 METHODS)")
print("="*70)

# Create a dummy video file
with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp4', delete=False) as f:
    f.write(b"This is a test video file content")
    test_video = f.name

print(f"\n   Created test video: {os.path.basename(test_video)}")

# Method A: FFmpeg Metadata (preferred)
print("\n   Method A: FFmpeg Metadata Watermarking")
print("   ----------------------------------------")
print("   ✅ Embeds data INSIDE video file (metadata tags)")
print("   ✅ Invisible to viewers")
print("   ✅ Fast (no re-encoding needed)")
print("   ⚠️  Requires ffmpeg installed")

# Method B: LSB Watermarking (future)
print("\n   Method B: LSB (Least Significant Bit) Watermarking")
print("   ---------------------------------------------------")
print("   ✅ Modifies pixel values imperceptibly")
print("   ✅ Survives compression better")
print("   ⏳ TODO: Not yet fully implemented")

# Method C: Sidecar file (fallback)
print("\n   Method C: Sidecar JSON File (.watermark.json)")
print("   ----------------------------------------------")
print("   ✅ Always works (no dependencies)")
print("   ❌ Separate file (can be deleted)")
print("   📝 Development/testing only")

# Embed watermark using current implementation
print(f"\n   Embedding watermark in test video...")
watermarked_video = embed_watermark(test_video, build_id="demo_build_20251106")
print(f"   ✅ Watermarked: {os.path.basename(watermarked_video)}")

# Check what files were created
if os.path.exists(watermarked_video + '.watermark.json'):
    print(f"   📄 Sidecar file created: {os.path.basename(watermarked_video)}.watermark.json")
    print("       (This is the fallback method - ffmpeg not available)")

# Demo 3: Watermark Detection
print("\n" + "="*70)
print("3️⃣  WATERMARK DETECTION")
print("="*70)

result = detect_watermark(watermarked_video)
if result and result.get('found'):
    print("\n   ✅ Watermark detected!")
    print(f"   Build ID: {result.get('build_id')}")
    print(f"   Method: {result.get('detection_method')}")
    print(f"   Timestamp: {result.get('watermarked_at', 'N/A')}")
else:
    print("\n   ❌ No watermark detected")

# Demo 4: What the video looks like
print("\n" + "="*70)
print("4️⃣  VISUAL IMPACT")
print("="*70)
print("\n   ❌ NO visible watermark image (no 'BHIV' logo overlay)")
print("   ❌ NO text watermark on video")
print("   ❌ NO branding overlay")
print("   ✅ Video looks EXACTLY the same to viewers")
print("   ✅ Watermark is INVISIBLE (metadata only)")

# Demo 5: Use case
print("\n" + "="*70)
print("5️⃣  REAL-WORLD USE CASE")
print("="*70)
print("""
   Scenario: Someone copies your video and shares it elsewhere

   1. You find the suspicious video file
   2. Run: python tools/detect_provenance.py suspicious_video.mp4
   3. Tool extracts the watermark and shows:
      - BUILD_ID: build_20251106_001
      - SHA256 hash: a1b2c3d4...
      - Created: 2025-11-06T10:30:00Z
   4. You check your BHIV registry:
      - Build ID matches your production build
      - Hash matches your original file
   5. Conclusion: ✅ This IS your video (proven!)
   
   The watermark is like a DNA test for your videos!
""")

# Cleanup
print("\n" + "="*70)
print("CLEANUP")
print("="*70)
print(f"\n   Removing test files...")
if os.path.exists(test_video):
    os.unlink(test_video)
    print(f"   ✅ Deleted: {os.path.basename(test_video)}")

if os.path.exists(watermarked_video):
    os.unlink(watermarked_video)
    print(f"   ✅ Deleted: {os.path.basename(watermarked_video)}")

watermark_json = watermarked_video + '.watermark.json'
if os.path.exists(watermark_json):
    os.unlink(watermark_json)
    print(f"   ✅ Deleted: {os.path.basename(watermark_json)}")

print("\n" + "="*70)
print("✅ DEMO COMPLETE")
print("="*70)
print("""
Key Takeaways:
1. Watermark is INVISIBLE (no visual impact on video)
2. Uses BUILD_ID to create deterministic fingerprint
3. Three methods: FFmpeg metadata (best), LSB (future), Sidecar (fallback)
4. Can be detected to prove ownership
5. Works like DNA testing for videos
""")
