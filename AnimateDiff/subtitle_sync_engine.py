#!/usr/bin/env python3
"""
Advanced Subtitle Synchronization Engine
Auto-generates .srt from TTS with precise timing
Supports Devanagari/Hindi/English fonts for Gurukul styling
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Protocol, Sequence
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
import logging

logger = logging.getLogger(__name__)

class AudioClipProtocol(Protocol):
    """Protocol for audio clips that have duration attribute"""
    duration: float

class SubtitleSyncEngine:
    """Advanced subtitle engine with TTS timing synchronization"""
    
    def __init__(self):
        self.gurukul_fonts = {
            'devanagari': 'Noto Sans Devanagari',
            'hindi': 'Mangal',
            'english': 'Arial',
            'sanskrit': 'Sanskrit Text'
        }
        
    def generate_precise_subtitles(self, audio_clips: Sequence[AudioClipProtocol], 
                                 text_segments: List[str], 
                                 output_path: str,
                                 language: str = 'english') -> str:
        """
        Generate .srt file with precise timing from actual audio clips
        
        Args:
            audio_clips: List of audio clips with actual durations
            text_segments: Corresponding text for each audio clip
            output_path: Path for output .srt file
            language: Language for font selection
        
        Returns:
            Path to generated .srt file
        """
        
        try:
            logger.info(f"Generating precise subtitles for {len(audio_clips)} segments")
            
            srt_content = []
            current_time = 0.0
            
            for i, (audio_clip, text) in enumerate(zip(audio_clips, text_segments)):
                # Get actual audio duration for precise timing
                duration = audio_clip.duration
                
                start_time = current_time
                end_time = current_time + duration
                
                # Format times for SRT
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                # Clean and format text
                clean_text = self._clean_text_for_subtitles(text)
                
                # Add subtitle entry
                srt_entry = f"{i + 1}\n{start_srt} --> {end_srt}\n{clean_text}\n"
                srt_content.append(srt_entry)
                
                current_time = end_time
                
                logger.debug(f"Subtitle {i+1}: {start_srt} - {end_srt} | {clean_text[:50]}...")
            
            # Write SRT file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(srt_content))
            
            logger.info(f"Subtitles generated: {output_path}")
            logger.info(f"Total duration: {current_time:.2f}s, Segments: {len(audio_clips)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Subtitle generation failed: {e}")
            raise
    
    def embed_subtitles_in_video(self, video_path: str, srt_path: str, 
                               output_path: str, language: str = 'english',
                               style: str = 'gurukul') -> str:
        """
        Embed subtitles directly into video using ffmpeg with Gurukul styling
        
        Args:
            video_path: Input video file
            srt_path: Subtitle file (.srt)
            output_path: Output video with embedded subtitles
            language: Language for font selection
            style: Subtitle style (gurukul, modern, classic)
        
        Returns:
            Path to video with embedded subtitles
        """
        
        try:
            logger.info(f"Embedding subtitles: {srt_path} -> {video_path}")
            
            # Get font for language
            font_name = self.gurukul_fonts.get(language, 'Arial')
            
            # Perfect movie-style subtitle parameters
            if style == 'gurukul':
                font_size = 28  # Reduced size as requested
                font_color = 'FFFFFF'  # Bright white
                outline_color = '000000'  # Minimal black outline
                outline_width = 1  # Very thin outline
                background_color = 'transparent'  # No background for cleaner look
                position = 'bottom'
            elif style == 'modern':
                font_size = 26  # Reduced size
                font_color = 'FFFFFF'  # Bright white
                outline_color = '000000'  # Minimal outline
                outline_width = 1
                background_color = 'transparent'
                position = 'bottom'
            else:  # classic
                font_size = 24  # Reduced size
                font_color = 'FFFFFF'  # Bright white
                outline_color = '000000'  # Minimal outline
                outline_width = 1
                background_color = 'transparent'
                position = 'bottom'
            
            # Build ffmpeg command for subtitle embedding with perfect positioning
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', video_path,  # Input video
                '-vf', f"subtitles={srt_path}:force_style='FontName={font_name},FontSize={font_size},PrimaryColour=&H{self._color_to_hex(font_color)},OutlineColour=&H{self._color_to_hex(outline_color)},BorderStyle=1,Outline={outline_width},BackColour=&H{self._color_to_hex(background_color)},Alignment=2,MarginV=60'",  # Added MarginV for bottom margin
                '-c:a', 'copy',  # Copy audio without re-encoding
                output_path
            ]
            
            logger.info(f"Running ffmpeg subtitle embedding...")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info(f"Subtitles embedded successfully: {output_path}")
                return output_path
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                # Fallback to MoviePy method
                return self._embed_subtitles_moviepy(video_path, srt_path, output_path, language)
                
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg subtitle embedding timed out")
            return self._embed_subtitles_moviepy(video_path, srt_path, output_path, language)
        except Exception as e:
            logger.error(f"Subtitle embedding failed: {e}")
            return self._embed_subtitles_moviepy(video_path, srt_path, output_path, language)
    
    def _embed_subtitles_moviepy(self, video_path: str, srt_path: str, 
                               output_path: str, language: str) -> str:
        """Fallback subtitle embedding using MoviePy"""
        
        try:
            logger.info("Using MoviePy fallback for subtitle embedding")
            
            # Load video
            video = VideoFileClip(video_path)
            
            # Parse SRT file
            subtitle_clips = self._parse_srt_to_clips(srt_path, video.w, language)
            
            # Composite video with subtitles
            if subtitle_clips:
                final_video = CompositeVideoClip([video] + subtitle_clips)
            else:
                final_video = video
            
            # Write output
            final_video.write_videofile(
                output_path,
                fps=video.fps,
                verbose=False,
                logger=None,
                codec='libx264',
                audio_codec='aac'
            )
            
            # Cleanup
            video.close()
            final_video.close()
            for clip in subtitle_clips:
                clip.close()
            
            logger.info(f"MoviePy subtitle embedding completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"MoviePy subtitle embedding failed: {e}")
            # Return original video if all subtitle methods fail
            return video_path
    
    def _parse_srt_to_clips(self, srt_path: str, video_width: int, language: str) -> List[TextClip]:
        """Parse SRT file and create TextClip objects"""
        
        subtitle_clips = []
        font_name = self.gurukul_fonts.get(language, 'Arial')
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse SRT format
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for match in matches:
                index, start_time, end_time, text = match
                
                # Convert time to seconds
                start_seconds = self._srt_time_to_seconds(start_time)
                end_seconds = self._srt_time_to_seconds(end_time)
                duration = end_seconds - start_seconds
                
                # Clean text
                clean_text = text.strip().replace('\n', ' ')
                
                if clean_text and duration > 0:
                    # FIXED: Movie-style subtitle positioning (moved significantly up)
                    text_clip = TextClip(
                        clean_text,
                        fontsize=30,  # Reduced size as requested
                        color='white',  # Pure white
                        font=font_name,
                        method='caption',  # Removed stroke completely
                        size=(video_width * 0.85, None),
                        align='center'
                    ).set_position(('center', 350)).set_start(start_seconds).set_duration(duration)  # FIXED: 350px from top (much higher)
                    
                    subtitle_clips.append(text_clip)
            
            logger.info(f"Parsed {len(subtitle_clips)} subtitle clips from SRT")
            return subtitle_clips
            
        except Exception as e:
            logger.error(f"SRT parsing failed: {e}")
            return []
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"
    
    def _srt_time_to_seconds(self, srt_time: str) -> float:
        """Convert SRT time format to seconds"""
        time_part, ms_part = srt_time.split(',')
        h, m, s = map(int, time_part.split(':'))
        ms = int(ms_part)
        
        return h * 3600 + m * 60 + s + ms / 1000.0
    
    def _clean_text_for_subtitles(self, text: str) -> str:
        """Clean and format text for subtitles"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Ensure proper sentence ending
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Break long lines for better readability
        if len(text) > 60:
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= 60:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            text = '\n'.join(lines)
        
        return text
    
    def _color_to_hex(self, color: str) -> str:
        """Convert color name to hex for ffmpeg"""
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'yellow': 'FFFF00',
            'red': 'FF0000',
            'blue': '0000FF',
            'green': '00FF00',
            'transparent': '00000000'
        }
        
        return color_map.get(color.lower(), 'FFFFFF')
    
    def generate_subtitle_report(self, srt_path: str) -> Dict:
        """Generate detailed report about subtitle timing and content"""
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse subtitles
            pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\d+\n|\n*$)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            total_duration = 0
            total_chars = 0
            segments = []
            
            for match in matches:
                index, start_time, end_time, text = match
                start_seconds = self._srt_time_to_seconds(start_time)
                end_seconds = self._srt_time_to_seconds(end_time)
                duration = end_seconds - start_seconds
                
                clean_text = text.strip().replace('\n', ' ')
                char_count = len(clean_text)
                
                segments.append({
                    'index': int(index),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'text': clean_text,
                    'char_count': char_count,
                    'reading_speed': char_count / duration if duration > 0 else 0
                })
                
                total_duration += duration
                total_chars += char_count
            
            report = {
                'total_segments': len(segments),
                'total_duration': total_duration,
                'total_characters': total_chars,
                'average_reading_speed': total_chars / total_duration if total_duration > 0 else 0,
                'segments': segments,
                'file_path': srt_path,
                'generated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Subtitle report: {len(segments)} segments, {total_duration:.1f}s total")
            return report
            
        except Exception as e:
            logger.error(f"Subtitle report generation failed: {e}")
            return {}

if __name__ == "__main__":
    # Test subtitle sync engine
    engine = SubtitleSyncEngine()
    
    # Test SRT generation
    test_segments = [
        "This is the first subtitle segment.",
        "Here comes the second segment with more text.",
        "And finally the third segment to complete the test."
    ]
    
    # Mock audio clips for testing
    class MockAudioClip:
        def __init__(self, duration: float) -> None:
            self.duration: float = duration
    
    test_audio_clips = [
        MockAudioClip(3.5),
        MockAudioClip(4.2),
        MockAudioClip(3.8)
    ]
    
    srt_path = engine.generate_precise_subtitles(
        test_audio_clips, 
        test_segments, 
        "test_subtitles.srt"
    )
    
    print(f"Test SRT generated: {srt_path}")
    
    # Generate report
    report = engine.generate_subtitle_report(srt_path)
    print(f"Subtitle report: {report['total_segments']} segments, {report['total_duration']:.1f}s")
