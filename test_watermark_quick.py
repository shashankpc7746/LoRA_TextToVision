"""Quick test of watermarking on an existing video"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("QUICK WATERMARK TEST")
print("="*70)

# Test video path
test_video = r"AnimateDiff\outputs\multi_clip\The_Ancient_Wisdom_of_Cosmic_Consciousness_realistic_complete.mp4"

if not os.path.exists(test_video):
    print(f"❌ Test video not found: {test_video}")
    sys.exit(1)

print(f"\n✅ Found test video: {os.path.basename(test_video)}")
print(f"   Size: {os.path.getsize(test_video) / (1024*1024):.2f} MB")

# Import security modules
print("\n📦 Importing security modules...")
try:
    from security import embed_watermark, compute_fingerprint
    from security.visible_watermark import add_visible_watermark
    print("   ✅ Security modules imported")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 1: Add invisible watermark
print("\n💧 Test 1: Adding invisible watermark...")
try:
    build_id = "test_watermark_20251108"
    output_invisible = test_video.replace('.mp4', '_watermarked_invisible.mp4')
    
    result = embed_watermark(
        test_video,
        build_id=build_id,
        output_path=output_invisible
    )
    
    if result and os.path.exists(result):
        print(f"   ✅ Invisible watermark added!")
        print(f"   📁 Output: {os.path.basename(result)}")
        
        # Verify metadata was added
        import subprocess
        check_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format_tags', result]
        check_result = subprocess.run(check_cmd, capture_output=True, text=True)
        if 'BHIV' in check_result.stdout or 'BlackHole' in check_result.stdout:
            print(f"   ✅ Metadata verified in output!")
        else:
            print(f"   ⚠️  Metadata not found in output")
            print(f"   Output: {check_result.stdout}")
    else:
        print(f"   ❌ Invisible watermark failed - no output file")
        
except Exception as e:
    print(f"   ❌ Invisible watermark failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Add visible watermark
print("\n🎨 Test 2: Adding visible watermark...")
try:
    if 'result' in locals() and result:
        output_visible = test_video.replace('.mp4', '_watermarked_visible.mp4')
        
        visible_result = add_visible_watermark(
            result,  # Use the invisible watermarked version
            style="subtle",
            build_id=build_id
        )
        
        if visible_result and os.path.exists(visible_result):
            print(f"   ✅ Visible watermark added!")
            print(f"   📁 Output: {os.path.basename(visible_result)}")
            print(f"   💡 Open the video to verify BHI logo is centered")
        else:
            print(f"   ❌ Visible watermark failed - no output file")
    else:
        print(f"   ⏭️  Skipping (invisible watermark failed)")
        
except Exception as e:
    print(f"   ❌ Visible watermark failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Compute fingerprint
print("\n🔐 Test 3: Computing fingerprint...")
try:
    if 'visible_result' in locals() and visible_result:
        fingerprint = compute_fingerprint(visible_result)
        
        if fingerprint:
            print(f"   ✅ Fingerprint computed!")
            print(f"   SHA256: {fingerprint.get('sha256', 'N/A')[:32]}...")
            print(f"   BLAKE2b: {fingerprint.get('blake2b', 'N/A')[:32]}...")
        else:
            print(f"   ❌ Fingerprint computation failed")
    else:
        print(f"   ⏭️  Skipping (visible watermark failed)")
        
except Exception as e:
    print(f"   ❌ Fingerprint failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)

if 'visible_result' in locals() and visible_result and os.path.exists(visible_result):
    print(f"\n✅ SUCCESS! Watermarked video created:")
    print(f"   {visible_result}")
    print(f"\n📝 Next steps:")
    print(f"   1. Play the video to verify centered BHI logo")
    print(f"   2. Run: ffprobe -show_entries format_tags {os.path.basename(visible_result)}")
    print(f"   3. Verify metadata contains BHIV copyright info")
else:
    print(f"\n❌ Tests failed - check errors above")
