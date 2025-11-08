#!/usr/bin/env python3
"""Test what tags embed_watermark() actually creates"""
import subprocess
import json
import os
from security import embed_watermark

# Test video
test_video = r"AnimateDiff\storage\2025-11-08\The_Mountain's_Ancient_Wisdom_realistic_complete.mp4"

print("🧪 Testing embed_watermark() tag creation...\n")

# Create watermarked version
watermarked = embed_watermark(test_video, build_id='test_check_12345', output_path='test_watermark_check.mp4')
print(f"✅ Watermarked file created: {watermarked}\n")

# Extract tags with ffprobe
result = subprocess.run(
    ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', watermarked],
    capture_output=True, text=True
)

data = json.loads(result.stdout)
tags = data.get('format', {}).get('tags', {})

print(f"📋 Tags in watermarked file ({len(tags)} total):\n")
for k, v in tags.items():
    if len(str(v)) > 80:
        print(f"  • {k}: {str(v)[:80]}...")
    else:
        print(f"  • {k}: {v}")

# Check for critical tags
print("\n🔍 Critical watermark tags:")
if 'BHIV_WATERMARK' in tags:
    print(f"  ✅ BHIV_WATERMARK: Present (length: {len(tags['BHIV_WATERMARK'])})")
else:
    print(f"  ❌ BHIV_WATERMARK: MISSING!")

if 'BUILD_ID' in tags:
    print(f"  ✅ BUILD_ID: {tags['BUILD_ID']}")
else:
    print(f"  ❌ BUILD_ID: MISSING!")

# Cleanup
if os.path.exists(watermarked):
    os.remove(watermarked)
    print(f"\n🧹 Cleaned up test file")
