import os
import logging
import json
import argparse
from typing import Dict, List
from datetime import datetime, timezone
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv
import time
import re
import unicodedata

# Load environment variables
load_dotenv()

# Logging setup
USER_ID = 'agent_t_translator'
logger = logging.getLogger('translation_agent')
logger.setLevel(logging.DEBUG)  # Enable DEBUG level
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'translation_agent.txt'), encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  # Capture DEBUG logs
formatter = logging.Formatter('%(asctime)s - %(levelname)s - User: %(user)s - %(message)s')
file_handler.setFormatter(formatter)
file_handler.addFilter(lambda record: setattr(record, 'user', USER_ID) or True)
logger.handlers = [file_handler]
logger.info("Initializing translation_agent.py")

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GOOGLE_GEMINI_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

# Constants
INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'content', 'multilingual_ready')
METADATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'content', 'structured', 'metadata')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
LANGUAGE_MAP = {
    'en': ('English', 'Latin'),
    'hi': ('Hindi', 'Devanagari'),
    'sa': ('Sanskrit', 'Devanagari'),
    'mr': ('Marathi', 'Devanagari'),
    'ta': ('Tamil', 'Tamil'),
    'te': ('Telugu', 'Telugu'),
    'kn': ('Kannada', 'Kannada'),
    'ml': ('Malayalam', 'Malayalam'),
    'gu': ('Gujarati', 'Gujarati'),
    'bn': ('Bengali', 'Bengali'),
    'pa': ('Punjabi', 'Gurmukhi'),
    'es': ('Spanish', 'Latin'),
    'fr': ('French', 'Latin'),
    'de': ('German', 'Latin'),
    'zh': ('Chinese', 'CJK'),
    'ja': ('Japanese', 'CJK'),
    'ru': ('Russian', 'Cyrillic'),
    'ar': ('Arabic', 'Arabic'),
    'pt': ('Portuguese', 'Latin'),
    'it': ('Italian', 'Latin')
}

def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    logger.debug(f"Loading JSON: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.debug(f"Loaded JSON content: {data}")
            return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {str(e)}")
        raise

def validate_script(text: str, expected_script: str) -> bool:
    """Validate if text matches the expected script."""
    if not text or text.startswith('['):
        return False
    if expected_script == 'CJK':
        for char in text:
            name = unicodedata.name(char, '')
            if 'CJK' in name or 'HIRAGANA' in name or 'KATAKANA' in name:
                return True
        logger.warning(f"Text '{text}' does not match expected script {expected_script}")
        return False
    for char in text:
        if unicodedata.name(char, '').startswith(expected_script.upper()):
            return True
    logger.warning(f"Text '{text}' does not match expected script {expected_script}")
    return False

def translate_text_with_gemini(text: str, target_lang: str, source_lang: str = 'en') -> tuple[str, float]:
    """Translate text using Gemini API with retry logic."""
    lang_name, script = LANGUAGE_MAP[target_lang]
    logger.debug(f"Translating text from {source_lang} to {lang_name} ({target_lang}) using Gemini API")
    
    max_retries = 3
    base_backoff_seconds = 5

    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = (
                f"Translate the following text from {source_lang} to {lang_name} (language code: {target_lang}). "
                f"Ensure the translation uses the {script} script where applicable. "
                f"Provide the translation and a confidence score (a float between 0.0 and 1.0). "
                f"Return the output as a JSON object with two keys: 'translated_text' and 'confidence_score'.\n\n"
                f"Text to translate: \"{text}\""
            )
            
            response = model.generate_content(prompt)
            cleaned_response_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            logger.debug(f"Gemini API raw response for {target_lang} (attempt {attempt + 1}): {cleaned_response_text}")
            
            try:
                translation_data = json.loads(cleaned_response_text)
                translated_text = translation_data.get('translated_text', f"[{target_lang.upper()}] Error: Could not parse translation.")
                confidence = float(translation_data.get('confidence_score', 0.0))
                
                # Validate script for non-Latin languages
                if script != 'Latin' and not validate_script(translated_text, script):
                    logger.error(f"Invalid script for {lang_name} ({target_lang}): {translated_text}")
                    translated_text = f"[{target_lang.upper()}] Error: Incorrect script."
                    confidence = 0.0
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON for {target_lang} (attempt {attempt + 1}): {e}. Response: {cleaned_response_text}")
                translated_text = f"[{target_lang.upper()}] Error: Malformed JSON response from API."
                confidence = 0.0
            except (TypeError, ValueError) as e:
                logger.error(f"Error processing confidence score for {target_lang} (attempt {attempt + 1}): {e}. Response: {cleaned_response_text}")
                translated_text = translation_data.get('translated_text', f"[{target_lang.upper()}] Error: Could not parse translation.")
                confidence = 0.0

            return translated_text, confidence

        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit for {target_lang} on attempt {attempt + 1}/{max_retries}. Error: {str(e)}")
            if attempt < max_retries - 1:
                retry_delay_seconds = base_backoff_seconds * (2 ** attempt)
                match = re.search(r"retry_delay {\s*seconds: (\d+)\s*}", str(e), re.IGNORECASE)
                if match:
                    suggested_delay = int(match.group(1))
                    retry_delay_seconds = max(suggested_delay, retry_delay_seconds)
                    logger.info(f"API suggested retry_delay of {suggested_delay}s. Using {retry_delay_seconds}s.")
                
                logger.info(f"Retrying for {target_lang} in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                logger.error(f"Max retries reached for {target_lang}. API call failed. Error: {str(e)}")
                return f"[{target_lang.upper()}] Error: API call failed after {max_retries} retries (Rate Limit).", 0.0
        except Exception as e:
            logger.error(f"Error during Gemini API call for {target_lang} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(base_backoff_seconds)
                continue
            return f"[{target_lang.upper()}] Error: API call failed.", 0.0
    
    logger.error(f"Exited retry loop without success for {target_lang}.")
    return f"[{target_lang.upper()}] Error: API call failed after all retries.", 0.0

def get_mapped_metadata(content_id: str, platform: str) -> Dict:
    """Load voice_tag and other metadata from language_mapper.py output."""
    metadata_file = os.path.join(METADATA_DIR, f"metadata_{content_id}_{platform}_mapped.json")
    logger.debug(f"Attempting to load metadata file: {metadata_file}")
    if not os.path.exists(metadata_file):
        logger.warning(f"Metadata file not found: {metadata_file}. Using defaults.")
        return {}
    try:
        metadata = load_json(metadata_file)
        logger.info(f"Successfully loaded metadata for content_id: {content_id}, platform: {platform}. Voice_tag: {metadata.get('voice_tag', 'not found')}")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata {metadata_file}: {str(e)}")
        return {}

def translate_content(content_id: str, platform: str) -> List[Dict]:
    """Translate content for all target languages."""
    logger.info(f"Translating content_id: {content_id} for platform: {platform}")
    input_file = None
    for pipeline in ['latin_pipeline', 'devanagari_pipeline', 'dravidian_pipeline', 'cjk_pipeline', 'arabic_pipeline']:
        file_path = os.path.join(INPUT_DIR, pipeline, f"{content_id}_{platform}.json")
        if os.path.exists(file_path):
            input_file = file_path
            break
    
    if not input_file:
        logger.error(f"No input file found for content_id {content_id} and platform {platform}")
        raise FileNotFoundError(f"No input file found for content_id {content_id} and platform {platform}")
    
    try:
        content = load_json(input_file)
        metadata = get_mapped_metadata(content_id, platform)
        original_text = content.get('post', '')
        if not original_text.strip():
            logger.error(f"Empty text for content_id {content_id} and platform {platform}")
            raise ValueError(f"Empty text for content_id {content_id}")
        source_language = content.get('source_language', 'en')
        sentiment = content.get('sentiment', 'neutral')
        preferred_tone = content.get('preferred_tone', 'neutral')
        pipeline = content.get('pipeline', 'latin_pipeline')
        voice_tag = metadata.get('voice_tag', f"{source_language}_default")
        logger.debug(f"Using voice_tag: {voice_tag} for content_id: {content_id}, platform: {platform}")
        translations = []
        
        for lang_idx, lang in enumerate(LANGUAGE_MAP.keys()):
            if lang == source_language:
                translated_text = original_text
                confidence = 1.0
            else:
                translated_text, confidence = translate_text_with_gemini(original_text, lang, source_lang=source_language)
                if lang_idx < len(LANGUAGE_MAP) - 1:
                    time.sleep(1)
            
            translation_entry = {
                'content_id': content_id,
                'platform': platform,
                'language': lang,
                'translated_text': translated_text,
                'confidence_score': confidence,
                'source_language': source_language,
                'sentiment': sentiment,
                'preferred_tone': preferred_tone,
                'pipeline': pipeline,
                'voice_tag': voice_tag,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            translations.append(translation_entry)
            logger.info(f"Translated to {lang} with confidence {confidence}")
        
        return translations
    except Exception as e:
        logger.error(f"Failed to translate content_id {content_id}: {str(e)}")
        raise

def save_translations(translations: List[Dict], content_id: str, platform: str) -> None:
    """Save translations to a JSON file, appending to existing data."""
    output_file = os.path.join(OUTPUT_DIR, f"translated_content_{content_id}_{platform}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_translations = []
    
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_translations = json.load(f)
            logger.info(f"Loaded existing translations from {output_file}")
        
        # Append new translations, avoiding duplicates
        existing_keys = {(t['content_id'], t['platform'], t['language']) for t in existing_translations}
        new_translations = [t for t in translations if (t['content_id'], t['platform'], t['language']) not in existing_keys]
        existing_translations.extend(new_translations)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_translations, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved translations to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save translations to {output_file}: {str(e)}")
        raise

def generate_video_metadata(content_id: str, platform: str, language: str, voice_tag: str = None) -> Dict:
    """Generate metadata for video generation.
    
    Args:
        content_id: Unique identifier for the content
        platform: Platform where the content will be published
        language: Target language code
        voice_tag: Optional voice tag to use for TTS
        
    Returns:
        Dictionary containing video metadata
    """
    logger.debug(f"Generating video metadata for content_id: {content_id}, platform: {platform}, language: {language}")
    
    if not voice_tag:
        voice_tag = f"{language}_default"
    
    lang_name, script = LANGUAGE_MAP.get(language, ('Unknown', 'Latin'))
    
    metadata = {
        'content_id': content_id,
        'platform': platform,
        'language': language,
        'language_name': lang_name,
        'script': script,
        'voice_tag': voice_tag,
        'video_format': 'mp4',
        'timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    # Create output directory if it doesn't exist
    metadata_path = os.path.join(METADATA_DIR, f"video_metadata_{content_id}_{platform}_{language}.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    # Save metadata to file
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Generated video metadata for content_id: {content_id}, platform: {platform}, language: {language}")
    return metadata

def get_video_metadata(content_id: str, platform: str, language: str) -> Dict:
    """Retrieve video metadata if it exists, or generate new metadata.
    
    Args:
        content_id: Unique identifier for the content
        platform: Platform where the content will be published
        language: Target language code
        
    Returns:
        Dictionary containing video metadata
    """
    metadata_path = os.path.join(METADATA_DIR, f"video_metadata_{content_id}_{platform}_{language}.json")
    logger.debug(f"Attempting to load video metadata: {metadata_path}")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Successfully loaded video metadata for content_id: {content_id}, platform: {platform}, language: {language}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load video metadata {metadata_path}: {str(e)}")
    
    # If metadata doesn't exist or couldn't be loaded, generate new metadata
    return generate_video_metadata(content_id, platform, language)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Vaani Sentinel X: Translation Agent")
    parser.add_argument('--content_id', required=True, help="Content ID to translate")
    parser.add_argument('--platform', default='instagram', choices=['twitter', 'instagram', 'linkedin', 'sanatan'], help="Platform")
    args = parser.parse_args()
    
    translations = translate_content(args.content_id, args.platform)
    save_translations(translations, args.content_id, args.platform)

if __name__ == "__main__":
    main()
