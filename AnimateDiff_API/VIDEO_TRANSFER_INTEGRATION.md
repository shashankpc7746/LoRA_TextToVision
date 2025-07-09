# Video Transfer Integration for AnimateDiff System

## 🎯 Overview

This integration allows your AnimateDiff system (192.168.0.121:8501) to automatically send generated videos to the main production system (192.168.0.121:8001) for display and management.

## 🔧 What Was Added

### 1. **Enhanced FastAPI Backend** (`main.py`)

#### New Endpoints:
- **`/generate-video-with-transfer`** - Generates video and returns JSON with transfer status
- **`/send-video-to-main`** - Manual endpoint to send existing videos to main system
- **`/generate-video`** - Updated to automatically transfer videos (maintains backward compatibility)

#### New Features:
- Automatic video transfer to main system after generation
- Comprehensive error handling for network issues
- Detailed logging of transfer operations
- Support for video metadata (subject, topic, prompt, generation parameters)

### 2. **Enhanced Streamlit UI** (`streamlit_app.py`)

#### New Features:
- **Subject & Topic Fields** - Users can specify video categories
- **Transfer Status Display** - Shows success/failure of video transfer
- **Video ID & Access URL** - Displays main system video information
- **Enhanced Progress Tracking** - Shows transfer progress
- **Fallback Handling** - Graceful handling when transfer fails

### 3. **Configuration**

#### Main System Settings:
```python
MAIN_SYSTEM_URL = "http://192.168.0.121:8001"
MAIN_SYSTEM_ENDPOINT = f"{MAIN_SYSTEM_URL}/receive-video"
```

#### API Authentication:
- Uses existing API key: `shashank_ka_vision786`
- Consistent authentication across all endpoints

## 🚀 How It Works

### Automatic Transfer Flow:
1. User submits video generation request via Streamlit UI
2. AnimateDiff generates the video locally
3. System automatically sends video to main system via POST request
4. Main system receives video with metadata
5. User sees transfer status and video ID in UI
6. Video is available on both local and main systems

### Manual Transfer Flow:
1. Use `/send-video-to-main` endpoint
2. Specify existing video file path
3. Add subject, topic, and prompt information
4. System transfers video to main system

## 📡 API Integration Details

### Video Transfer Request Format:
```python
# Multipart form data
files = {
    'video': ('generated_video.mp4', video_file, 'video/mp4')
}

data = {
    'metadata': json.dumps({
        "subject": "Video Subject",
        "topic": "Video Topic", 
        "prompt": "Generation prompt",
        "generated_at": "2025-07-08T13:30:00",
        "file_size": 1234567,
        "system_info": "AnimateDiff_192.168.0.121:8501",
        "num_frames": 32,
        "guidance_scale": 8.0,
        "steps": 25,
        "seed": 333,
        "fps": 8
    })
}

headers = {
    'x-api-key': 'shashank_ka_vision786'
}
```

### Expected Response from Main System:
```json
{
    "video_id": "unique_video_id",
    "access_url": "http://main-system/video/unique_video_id",
    "status": "success",
    "message": "Video received successfully"
}
```

## 🎮 Usage Instructions

### For End Users (Streamlit UI):
1. Open `http://localhost:8501`
2. Fill in video generation parameters
3. Add **Subject** and **Topic** for categorization
4. Click "Generate Video"
5. Monitor progress including transfer status
6. View video locally and get main system access info

### For Developers (API):
```python
# Generate video with automatic transfer
response = requests.post(
    "http://localhost:8002/generate-video-with-transfer",
    json={
        "prompt": "anime girl walking",
        "subject": "Animation",
        "topic": "Character Movement",
        # ... other parameters
    },
    headers={"x-api-key": "shashank_ka_vision786"}
)

result = response.json()
print(f"Video ID: {result['video_id']}")
print(f"Access URL: {result['access_url']}")
```

## 🔍 Testing

### Run Test Script:
```bash
cd AnimateDiff_API
python test_video_transfer.py
```

### Manual Testing:
1. **Health Check**: `GET http://localhost:8002/health`
2. **Generate & Transfer**: `POST http://localhost:8002/generate-video-with-transfer`
3. **Manual Transfer**: `POST http://localhost:8002/send-video-to-main`

## 🛠️ Configuration Options

### Change Main System URL:
Edit `main.py`:
```python
MAIN_SYSTEM_URL = "http://your-main-system:port"
```

### Disable Auto-Transfer:
Use the original `/generate-video` endpoint instead of `/generate-video-with-transfer`

### Custom Metadata:
Modify the `send_video_to_main_system()` function to include additional metadata fields

## 🚨 Error Handling

### Network Issues:
- Connection timeouts (60 second limit)
- Network unreachability
- Main system downtime

### Response Handling:
- Invalid API responses
- Authentication failures
- File transfer errors

### Fallback Behavior:
- Video generation continues even if transfer fails
- Local video file always available
- Error messages displayed to user
- Detailed logging for debugging

## 📊 Monitoring & Logs

### Console Output:
```
🎬 Sending video to main system: http://192.168.0.121:8001/receive-video
🎬 Video file: outputs/animation_20250708_133000.mp4
🎬 Metadata: {...}
✅ Video successfully sent to main system!
🎬 Video ID: vid_abc123
🎬 Access URL: http://192.168.0.121:8001/video/vid_abc123
```

### Error Logs:
```
❌ Failed to send video to main system: 500 - Internal Server Error
❌ Network error sending video to main system: Connection timeout
```

## 🔄 Backward Compatibility

- Original `/generate-video` endpoint still works
- Existing Streamlit functionality preserved
- No breaking changes to current workflows
- Optional transfer features

## 🎉 Benefits

1. **Seamless Integration** - Videos automatically appear in main system
2. **User-Friendly** - Clear status indicators and error messages
3. **Robust Error Handling** - Graceful fallbacks when transfer fails
4. **Comprehensive Metadata** - Rich information sent with each video
5. **Flexible Usage** - Both automatic and manual transfer options
6. **Production Ready** - Proper authentication and timeout handling

The integration is now complete and ready for production use! 🚀
