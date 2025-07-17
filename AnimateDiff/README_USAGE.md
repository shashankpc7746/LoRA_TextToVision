# 🎬 Unified Video Generation System - Usage Guide

## 📁 **Cleaned Up File Structure**

### **Main Files:**
- `unified_video_generator.py` - **NEW**: Complete video generation with audio & subtitles
- `multi_clip_generator.py` - Original video generation system (updated)
- `animate_gurukul.py` - Core animation functions

### **Lesson Files:**
- `lessons/lesson_space_adventure.json` - Space adventure story
- `lessons/lesson_ocean_adventure.json` - Ocean exploration story  
- `lessons/lesson_1_dharma.json` - Dharma teaching story
- `lessons/lesson_*.json` - Other lesson files

### **Output:**
- `outputs/multi_clip/` - All generated videos and clips

## 🚀 **How to Generate Videos**

### **Method 1: Unified System (RECOMMENDED)**
```bash
# Generate space adventure in realistic style with audio & subtitles
python multi_clip_generator.py realistic unified

# Generate in anime style
python multi_clip_generator.py anime unified

# Or run unified system directly
python unified_video_generator.py realistic 1
```

### **Method 2: Original System (Video Only)**
```bash
# Generate video clips only (no audio)
python multi_clip_generator.py realistic

# Generate in anime style
python multi_clip_generator.py anime
```

## 🎵 **Audio & Subtitle Features**

### **Speech Rate Options:**
- `1` - Normal speed (recommended)
- `0` - Slow speech
- `2` - Fast speech
- `3` - Very fast speech

### **Example with Custom Speech Rate:**
```bash
python unified_video_generator.py realistic 2  # Fast speech
```

## 📚 **Available Lessons**

1. **Space Adventure** (`lesson_space_adventure.json`)
   - Young astronaut's first mission
   - 8 scenes from preparation to return

2. **Ocean Adventure** (`lesson_ocean_adventure.json`)
   - Marine biologist exploration
   - 8 scenes of underwater discovery

3. **Dharma Teaching** (`lesson_1_dharma.json`)
   - Traditional teaching story
   - 8 scenes under banyan tree

## 🎨 **Available Styles**

- `realistic` - Photorealistic characters and environments
- `anime` - Traditional anime art style  
- `artistic` - Watercolor and oil painting effects

## 📊 **Output Files**

### **Unified System Generates:**
- `{LessonTitle}_{style}_complete.mp4` - Final video with audio & subtitles
- Individual clip files for debugging
- Audio files (temporary, auto-cleaned)

### **Original System Generates:**
- `final_video_NO_FADE_ENHANCED_CAMERA.mp4` - Video only
- Individual clip files
- Control images for debugging

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **No Audio**: Use unified system (`unified` parameter)
2. **Slow Audio**: Adjust speech rate parameter
3. **Subtitle Issues**: Check ImageMagick installation
4. **Generation Errors**: Check lesson file format

### **File Locations:**
- **Videos**: `outputs/multi_clip/`
- **Lessons**: `lessons/`
- **Logs**: Console output

## 🎯 **Quick Start**

```bash
# Best option - Complete video with audio and subtitles
python multi_clip_generator.py realistic unified

# This will generate:
# - Journey_to_the_Stars_realistic_complete.mp4
# - With synchronized audio narration
# - With synchronized subtitles
# - In realistic visual style
```

## 📝 **Notes**

- **Unified system** handles everything: video generation, audio creation, subtitle sync
- **Original system** only generates video clips (legacy)
- All temporary files are automatically cleaned up
- Videos are saved in `outputs/multi_clip/` directory
- Lesson files can be easily modified to create new stories

## 🚀 **What Was Consolidated**

### **Removed Files:**
- `fix_audio_timing.py` ❌
- `quick_fix_speed.py` ❌  
- `fix_audio_properly.py` ❌
- `clean_audio_integration.py` ❌
- `generate_space_video.py` ❌
- `add_audio_*.py` ❌ (multiple files)
- `simple_*.py` ❌ (multiple files)
- `working_subtitle_fix.py` ❌

### **Removed Folders:**
- `outputs/multi_clip_style_*` ❌ (empty style folders)

### **Consolidated Into:**
- `unified_video_generator.py` ✅ (All audio/video/subtitle functionality)
- Updated `multi_clip_generator.py` ✅ (Integration with unified system)

**Result: Much cleaner, easier to manage codebase!** 🎉
