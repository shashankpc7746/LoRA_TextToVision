from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gtts import gTTS
import uuid
import subprocess
import os
from pathlib import Path
import numpy as np
import librosa
import traceback
from keras.models import load_model
import json
from datetime import datetime, timezone

from translation_agent import translate_text_with_gemini, LANGUAGE_MAP  # Added LANGUAGE_MAP

app = FastAPI()

# Directories
TTS_OUTPUT_DIR = "tts/tts_outputs"
RESULTS_DIR = "results"
AVATAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "avatars"))
GENDER_MODEL_PATH = "gender-recognition-by-voice/results/model.h5"
WAV2LIP_PATH = "Wav2Lip"
WAV2LIP_CHECKPOINT = "checkpoints/wav2lip_gan.pth"

os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Avatar directory: {AVATAR_DIR}")

gender_model = load_model(GENDER_MODEL_PATH) if os.path.exists(GENDER_MODEL_PATH) else None

AVATARS = {
    "female": [os.path.join(AVATAR_DIR, "pht1.jpg"), os.path.join(AVATAR_DIR, "pht2.jpg")],
    "male": [os.path.join(AVATAR_DIR, "pht3.jpg"), os.path.join(AVATAR_DIR, "pht4.jpg")],
    "default": os.path.join(AVATAR_DIR, "pht1.jpg")
}

for gender, paths in AVATARS.items():
    if gender == "default":
        paths = [paths]
    for path in paths:
        if not os.path.isfile(path):
            print(f"⚠️ Missing avatar file: {path}")

def extract_features(file_path: str) -> np.ndarray:
    try:
        from scipy.io import wavfile
        sample_rate, X = wavfile.read(file_path)

        if X.ndim > 1:
            X = X[:, 0]

        X = X.astype(np.float32)
        X = X / np.max(np.abs(X), axis=0)

        fft_spectrum = np.fft.fft(X)
        magnitude = np.abs(fft_spectrum[:len(fft_spectrum)//2])
        mel = np.log1p(magnitude[:128])

        if mel.size < 128:
            mel = np.pad(mel, (0, 128 - mel.size), mode='constant')

        return mel
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
    prediction = gender_model.predict(features, verbose=0)[0]
    return "male" if prediction >= 0.5 else "female"

def select_avatar(gender: str) -> str:
    import random
    avatar_path = random.choice(AVATARS.get(gender, [AVATARS["default"]]))
    if not os.path.isfile(avatar_path):
        raise FileNotFoundError(f"Avatar file not found: {avatar_path}")
    print(f"Selected avatar path: {avatar_path}")
    return avatar_path

def convert_mp3_to_wav(mp3_path: str, wav_path: str):
    subprocess.run(["ffmpeg", "-y", "-i", mp3_path, wav_path], check=True)

def run_sadtalker(audio_path: str, image_path: str, output_path: str):
    subprocess.run([
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", os.path.dirname(output_path)
    ], cwd="SadTalker", check=True)


def generate_video_metadata(session_id: str, text: str, language: str, gender: str, avatar_path: str) -> dict:
    """Generate metadata for the created video.
    
    Args:
        session_id: Unique identifier for the session
        text: The text used for TTS
        language: Language code
        gender: Detected gender for avatar selection
        avatar_path: Path to the avatar image used
        
    Returns:
        Dictionary containing video metadata
    """
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
    
    # Save metadata to file
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Generated metadata for session: {session_id}")
    return metadata

@app.post("/api/generate-and-sync")
async def generate_and_sync(text: str = Form(...), target_lang: str = Form(default='en')):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    if len(text) > 500:
        text = text[:500]

    original_text = text
    if target_lang != "en":
        translated_text, confidence = translate_text_with_gemini(text, target_lang, source_lang='en')
        if not translated_text or confidence < 0.1:
            raise HTTPException(status_code=500, detail=f"Translation failed: {translated_text}")
        text = translated_text

    session_id = str(uuid.uuid4())
    mp3_filename = f"{session_id}.mp3"
    mp3_path = os.path.join(TTS_OUTPUT_DIR, mp3_filename)

    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(mp3_path)

        wav_path = os.path.join(TTS_OUTPUT_DIR, f"{session_id}.wav")
        convert_mp3_to_wav(mp3_path, wav_path)

        gender = predict_gender(wav_path)
        avatar_path = select_avatar(gender)

        output_video = os.path.join(RESULTS_DIR, f"{session_id}.mp4")
        run_sadtalker(wav_path, avatar_path, output_video)
        
        # Generate and save metadata
        metadata = generate_video_metadata(
            session_id=session_id,
            text=original_text if target_lang == 'en' else text,
            language=target_lang,
            gender=gender,
            avatar_path=avatar_path
        )

        return FileResponse(
            path=output_video,
            filename=f"lipsync_{session_id}.mp4",
            media_type="video/mp4"
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg/Wav2Lip failed: {e.stderr}")
    except Exception:
        print("Unexpected error:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Unexpected error occurred")

@app.get("/")
def root():
    return {"message": "TTS-LipSync-Translation API running"}

@app.get("/api/metadata/{session_id}")
async def get_metadata(session_id: str):
    """Retrieve metadata for a specific video session."""
    metadata_path = os.path.join(RESULTS_DIR, f"metadata_{session_id}.json")
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return JSONResponse(content=metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {str(e)}")
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.125", port=8001)

