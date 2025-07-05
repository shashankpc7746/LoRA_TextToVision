from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
import uuid
import subprocess
import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import traceback
from keras.models import load_model
import json
from datetime import datetime, timezone
from translation_agent import translate_text_with_gemini, LANGUAGE_MAP
import shutil
import time
import glob

app = FastAPI()

# Directories
TTS_OUTPUT_DIR = "tts/tts_outputs"
RESULTS_DIR = "results"
AVATAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "avatars"))
SADTALKER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SadTalker"))
GENDER_MODEL_PATH = "gender-recognition-by-voice/results/model.h5"

os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

gender_model = load_model(GENDER_MODEL_PATH) if os.path.exists(GENDER_MODEL_PATH) else None

# 🗂️ FILE MANAGEMENT CONFIGURATION
MAX_VIDEO_FILES = 5  # Keep only 5 most recent videos
MAX_AUDIO_FILES = 5  # Keep only 5 most recent audio files

AVATARS = {
    "female": [os.path.join(AVATAR_DIR, "pht1.jpg"), os.path.join(AVATAR_DIR, "pht2.jpg")],
    "male": [os.path.join(AVATAR_DIR, "pht3.jpg"), os.path.join(AVATAR_DIR, "pht4.jpg")],
    "default": [os.path.join(AVATAR_DIR, "pht1.jpg")]
}

def extract_features(file_path: str) -> np.ndarray:
    try:
        from scipy.io import wavfile
        sample_rate, X = wavfile.read(file_path)
        if X.ndim > 1:
            X = X[:, 0]
        X = X.astype(np.float32) / np.max(np.abs(X), axis=0)
        fft_spectrum = np.fft.fft(X)
        magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2])
        mel = np.log1p(magnitude[:128])
        return np.pad(mel, (0, 128 - mel.size), mode='constant')
    except Exception:
        print("Feature extraction error:", traceback.format_exc())
        return np.array([])

def predict_gender(audio_path: str) -> str:
    if gender_model is None:
        return "default"
    features = extract_features(audio_path)
    if features.size != 128:
        features = np.pad(features, (0, 128 - features.shape[0]), mode='constant')
    features = np.expand_dims(features, axis=0)
    try:
        prediction = float(gender_model.predict(features, verbose=0)[0])
        return "male" if prediction >= 0.5 else "female"
    except Exception:
        print("[ERROR] Gender prediction error. Using default.")
        return "default"

def select_avatar(gender: str) -> str:
    import random
    if gender not in AVATARS:
        gender = "default"
    avatar_list = AVATARS[gender]
    avatar_path = random.choice(avatar_list)
    if not os.path.isfile(avatar_path):
        raise FileNotFoundError(f"Avatar not found: {avatar_path}")
    return avatar_path

def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    mp3_path = os.path.normpath(mp3_path)
    wav_path = os.path.normpath(wav_path)
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)
    waited = 0.0
    while waited < 5:
        try:
            with sf.SoundFile(wav_path):
                return
        except:
            time.sleep(0.1)
            waited += 0.1
    raise RuntimeError(f"WAV file not readable: {wav_path}")

def cleanup_old_files():
    """🗂️ Clean up old files to keep only the most recent ones"""
    try:
        # Clean up video files (keep only MAX_VIDEO_FILES most recent)
        video_files = glob.glob(os.path.join(RESULTS_DIR, "*.mp4"))
        if len(video_files) > MAX_VIDEO_FILES:
            # Sort by modification time (newest first)
            video_files.sort(key=os.path.getmtime, reverse=True)
            files_to_delete = video_files[MAX_VIDEO_FILES:]

            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"[CLEANUP] Deleted old video: {os.path.basename(file_path)}")

                    # Also delete corresponding metadata file
                    session_id = os.path.splitext(os.path.basename(file_path))[0]
                    metadata_file = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
                    if os.path.exists(metadata_file):
                        os.remove(metadata_file)
                        print(f"[CLEANUP] Deleted metadata: metadata_{session_id}.json")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {file_path}: {e}")

        # Clean up audio files (keep only MAX_AUDIO_FILES most recent)
        audio_files = glob.glob(os.path.join(TTS_OUTPUT_DIR, "*.mp3")) + \
                     glob.glob(os.path.join(TTS_OUTPUT_DIR, "*.wav"))
        if len(audio_files) > MAX_AUDIO_FILES * 2:  # *2 because we have both mp3 and wav
            audio_files.sort(key=os.path.getmtime, reverse=True)
            files_to_delete = audio_files[MAX_AUDIO_FILES * 2:]

            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    print(f"[CLEANUP] Deleted old audio: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"[ERROR] Failed to delete {file_path}: {e}")

        print(f"[CLEANUP] File cleanup completed. Keeping {MAX_VIDEO_FILES} videos and {MAX_AUDIO_FILES} audio sets")

    except Exception as e:
        print(f"[ERROR] File cleanup failed: {e}")

def run_sadtalker(audio_path, image_path, output_path=None) -> str:
    result_root_dir = os.path.abspath("tts_module/results")
    python_executable = sys.executable  # ✅ Ensure same environment

    # Convert paths to absolute paths
    audio_path = os.path.abspath(audio_path)
    image_path = os.path.abspath(image_path)

    # ⚡ PERFORMANCE OPTIMIZATIONS for faster processing
    subprocess.run([
        python_executable, "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", result_root_dir,
        "--still",  # ⚡ Still mode - faster processing
        "--preprocess", "crop",  # ⚡ Simple crop preprocessing
        "--size", "256",  # ⚡ Smaller size for faster processing
        "--batch_size", "2",  # ⚡ Larger batch size
    ], cwd=SADTALKER_DIR, check=True)

    # Locate latest .mp4
    latest_file = None
    latest_time = 0

    # Ensure the result directory exists
    os.makedirs(result_root_dir, exist_ok=True)

    for root, _, files in os.walk(result_root_dir):
        for file in files:
            if file.endswith(".mp4"):
                path = os.path.join(root, file)
                t = os.path.getmtime(path)
                if t > latest_time:
                    latest_file = path
                    latest_time = t

    if not latest_file:
        raise FileNotFoundError("SadTalker did not generate a video.")

    return latest_file

def generate_video_metadata(session_id: str, text: str, language: str, gender: str, avatar_path: str) -> dict:
    lang_name, script = LANGUAGE_MAP.get(language, ('Unknown', 'Latin'))
    metadata = {
        'session_id': session_id,
        'language': language,
        'language_name': lang_name,
        'script': script,
        'text_length': len(text),
        'gender': gender,
        'avatar': os.path.basename(avatar_path),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'video_format': 'mp4'
    }
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata

@app.post("/api/generate-and-sync")
async def generate_and_sync(text: str = Form(...), target_lang: str = Form(default='en')):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # ⚡ PERFORMANCE: Limit text length for faster processing
    if len(text) > 200:  # Reduced from 500 to 200 for faster processing
        text = text[:200]
        print(f"[INFO] Text truncated to 200 characters for faster processing")

    original_text = text
    if target_lang != "en":
        translated_text, confidence = translate_text_with_gemini(text, target_lang, source_lang='en')
        if not translated_text or confidence < 0.1:
            raise HTTPException(status_code=500, detail=f"Translation failed: {translated_text}")
        text = translated_text

    session_id = str(uuid.uuid4())
    mp3_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}.mp3")
    wav_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}.wav")

    try:
        gTTS(text=text, lang='en', slow=False).save(mp3_path)
        convert_mp3_to_wav(mp3_path, wav_path)

        gender = predict_gender(wav_path)
        avatar_path = select_avatar(gender)
        generated_path = run_sadtalker(wav_path, avatar_path)

        final_output_path = os.path.join(RESULTS_DIR, f"{session_id}.mp4")
        shutil.copy2(generated_path, final_output_path)

        generate_video_metadata(session_id, original_text, target_lang, gender, avatar_path)

        # 🗂️ Clean up old files after successful generation
        cleanup_old_files()

        return FileResponse(
            path=final_output_path,
            filename=f"lipsync_{session_id}.mp4",
            media_type="video/mp4"
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg/SadTalker failed: {e.stderr}")
    except Exception:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.get("/")
def root():
    return {"message": "TTS-LipSync-Translation API running"}

@app.get("/api/metadata/{session_id}")
async def get_metadata(session_id: str):
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return JSONResponse(content=json.load(f))

@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    filepath = os.path.join(TTS_OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=filepath, filename=filename, media_type='audio/mpeg')

@app.get("/api/list-audio-files")
async def list_audio_files():
    files = [f for f in os.listdir(TTS_OUTPUT_DIR) if f.endswith('.mp3')]
    return {"audio_files": files, "count": len(files)}

@app.post("/api/cleanup-files")
async def manual_cleanup():
    """🗂️ Manual endpoint to trigger file cleanup"""
    try:
        cleanup_old_files()

        # Count remaining files
        video_count = len(glob.glob(os.path.join(RESULTS_DIR, "*.mp4")))
        audio_count = len(glob.glob(os.path.join(TTS_OUTPUT_DIR, "*.mp3"))) + \
                     len(glob.glob(os.path.join(TTS_OUTPUT_DIR, "*.wav")))

        return {
            "message": "File cleanup completed successfully",
            "remaining_videos": video_count,
            "remaining_audio_files": audio_count,
            "max_videos": MAX_VIDEO_FILES,
            "max_audio_files": MAX_AUDIO_FILES
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.121", port=8001)
