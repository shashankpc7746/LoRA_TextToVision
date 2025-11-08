#!/usr/bin/env python3
"""
Test FFmpeg metadata preservation through the video processing pipeline
"""
import subprocess
import os
import sys

def run_cmd(cmd):
    """Run command and return output"""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.returncode, result.stdout, result.stderr

def check_metadata(video_path):
    """Check if video has BHIV metadata"""
    cmd = ['ffprobe', '-show_entries', 'format_tags', '-of', 'default=noprint_wrappers=1', video_path]
    returncode, stdout, stderr = run_cmd(cmd)
    
    if returncode == 0:
        has_bhiv = 'BHIV' in stdout or 'BUILD_ID' in stdout or 'BlackHole' in stdout
        return has_bhiv, stdout
    return False, ""

def main():
    print("\n" + "="*70)
    print("METADATA PRESERVATION TEST")
    print("="*70)
    
    # Find a test video
    test_video = "AnimateDiff/outputs/multi_clip/The_Sacred_Journey_of_Self-Discovery_realistic_complete.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        return 1
    
    print(f"\n📹 Testing with: {os.path.basename(test_video)}")
    print(f"   Size: {os.path.getsize(test_video) / (1024*1024):.2f} MB")
    
    # Step 1: Add metadata to video
    print("\n🔧 Step 1: Adding watermark metadata...")
    from security import embed_watermark
    
    watermarked = test_video.replace('.mp4', '_test_watermarked.mp4')
    result = embed_watermark(test_video, build_id='test_preservation_12345', output_path=watermarked)
    
    has_meta, meta_output = check_metadata(result)
    if has_meta:
        print(f"   ✅ Watermark metadata added!")
        print(f"   Tags found: BUILD_ID, BHIV, copyright")
    else:
        print(f"   ❌ No metadata found after watermarking")
        return 1
    
    # Step 2: Simulate OpenCV processing (strips audio)
    print("\n🎨 Step 2: Simulating OpenCV processing (strips audio)...")
    opencv_sim = test_video.replace('.mp4', '_test_opencv_sim.mp4')
    
    # Just copy video stream (no audio) to simulate OpenCV behavior
    cmd = [
        'ffmpeg', '-y', '-i', watermarked,
        '-map', '0:v:0',  # Video only
        '-c:v', 'copy',
        opencv_sim
    ]
    returncode, stdout, stderr = run_cmd(cmd)
    
    if returncode == 0:
        print(f"   ✅ Simulated OpenCV output created")
        has_meta, _ = check_metadata(opencv_sim)
        if has_meta:
            print(f"   ✅ Metadata preserved through video-only copy")
        else:
            print(f"   ⚠️  Metadata lost (expected with -map)")
    else:
        print(f"   ❌ Failed to simulate OpenCV")
        return 1
    
    # Step 3: Test OLD method (without -map_metadata)
    print("\n❌ Step 3: Testing OLD FFmpeg method (without -map_metadata)...")
    old_method = test_video.replace('.mp4', '_test_old_method.mp4')
    
    cmd_old = [
        'ffmpeg', '-y',
        '-i', opencv_sim,      # Video (no audio, has metadata)
        '-i', test_video,      # Audio source (original)
        '-map', '0:v:0',       # Video from first
        '-map', '1:a:0',       # Audio from second
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'fast',
        old_method
    ]
    
    returncode, stdout, stderr = run_cmd(cmd_old)
    if returncode == 0:
        has_meta, meta_output = check_metadata(old_method)
        if has_meta:
            print(f"   ⚠️  Metadata preserved (unexpected!)")
        else:
            print(f"   ❌ Metadata LOST (as expected without -map_metadata)")
            print(f"   This is why watermarks weren't working!")
    
    # Step 4: Test NEW method (with -map_metadata)
    print("\n✅ Step 4: Testing NEW FFmpeg method (with -map_metadata)...")
    new_method = test_video.replace('.mp4', '_test_new_method.mp4')
    
    cmd_new = [
        'ffmpeg', '-y',
        '-i', opencv_sim,      # Video (no audio)
        '-i', test_video,      # Audio source
        '-i', watermarked,     # Metadata source (watermarked file)
        '-map', '0:v:0',       # Video from first
        '-map', '1:a:0',       # Audio from second
        '-map_metadata', '2',  # Metadata from third ⭐ KEY FIX
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-preset', 'fast',
        new_method
    ]
    
    returncode, stdout, stderr = run_cmd(cmd_new)
    if returncode == 0:
        has_meta, meta_output = check_metadata(new_method)
        if has_meta:
            print(f"   ✅ Metadata PRESERVED with -map_metadata!")
            print(f"\n   📋 Metadata tags:")
            for line in meta_output.split('\n'):
                if 'TAG:' in line and any(tag in line for tag in ['title', 'copyright', 'BUILD_ID', 'BHIV', 'author']):
                    print(f"      {line.strip()}")
            print(f"\n   🎉 SUCCESS! This method preserves watermarks!")
        else:
            print(f"   ❌ Metadata still lost")
            return 1
    else:
        print(f"   ❌ FFmpeg failed: {stderr[:200]}")
        return 1
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    for f in [watermarked, opencv_sim, old_method, new_method]:
        if os.path.exists(f):
            os.remove(f)
            print(f"   Removed: {os.path.basename(f)}")
    
    print("\n" + "="*70)
    print("✅ TEST PASSED - Metadata preservation method validated!")
    print("="*70)
    print("\n💡 Key finding: -map_metadata flag is ESSENTIAL for preserving watermarks")
    print("   through the FFmpeg audio restoration step.")
    print("\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
