#!/usr/bin/env python3
"""
Fallback Video Generator
Creates static image + audio videos when main generation fails
"""

import os
import json
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
from PIL import Image, ImageDraw, ImageFont
import tempfile

class FallbackGenerator:
    def __init__(self):
        self.output_dir = "outputs/multi_clip"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_static_image(self, text, width=512, height=512):
        """Create a static background image with text"""
        # Create a gradient background
        img = Image.new('RGB', (width, height), color='#2c3e50')
        draw = ImageDraw.Draw(img)
        
        # Add gradient effect
        for y in range(height):
            color_value = int(44 + (y / height) * 40)  # Gradient from dark to lighter
            draw.line([(0, y), (width, y)], fill=(color_value, color_value + 20, color_value + 40))
        
        # Add title text
        try:
            # Try to use a nice font
            font_size = 32
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Add main title
        title = "Gurukul Learning"
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 100), title, fill='white', font=font)
        
        # Add subtitle
        subtitle = "Educational Content"
        try:
            subtitle_font = ImageFont.truetype("arial.ttf", 20)
        except:
            subtitle_font = ImageFont.load_default()
            
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        draw.text((subtitle_x, 150), subtitle, fill='#bdc3c7', font=subtitle_font)
        
        # Add decorative elements
        # Draw a simple border
        border_color = '#34495e'
        draw.rectangle([10, 10, width-10, height-10], outline=border_color, width=3)
        
        # Add some decorative circles
        for i in range(3):
            x = 50 + i * 150
            y = 300
            draw.ellipse([x-20, y-20, x+20, y+20], outline='#3498db', width=2)
        
        return img
    
    def create_fallback_video(self, lesson_data, output_filename, duration=30):
        """Create a fallback video with static image and audio"""
        try:
            lesson_title = lesson_data.get('title', 'Educational Content')
            lesson_text = lesson_data.get('text', 'Content not available')
            
            print(f"🔄 Creating fallback video for: {lesson_title}")
            
            # Create static background image
            static_img = self.create_static_image(lesson_title)
            
            # Save temporary image
            temp_img_path = tempfile.mktemp(suffix='.png')
            static_img.save(temp_img_path)
            
            # Create image clip
            img_clip = ImageClip(temp_img_path, duration=duration)
            
            # Create text overlay with lesson content
            try:
                # Split text into manageable chunks
                words = lesson_text.split()
                if len(words) > 20:
                    display_text = ' '.join(words[:20]) + "..."
                else:
                    display_text = lesson_text
                
                # Create text clip
                text_clip = TextClip(
                    display_text,
                    fontsize=24,
                    color='white',
                    font='Arial',
                    size=(450, None),
                    method='caption'
                ).set_position(('center', 'bottom')).set_duration(duration).set_margin(20)
                
                # Composite video with text
                final_clip = CompositeVideoClip([img_clip, text_clip])
                
            except Exception as text_error:
                print(f"⚠️ Text overlay failed: {text_error}")
                # Use just the image if text fails
                final_clip = img_clip
            
            # Generate simple audio if TTS is enabled
            audio_clip = None
            if lesson_data.get('tts', False):
                try:
                    audio_clip = self.create_simple_audio(lesson_text, duration)
                    if audio_clip:
                        final_clip = final_clip.set_audio(audio_clip)
                except Exception as audio_error:
                    print(f"⚠️ Audio generation failed: {audio_error}")
            
            # Write video file
            output_path = os.path.join(self.output_dir, output_filename)
            final_clip.write_videofile(
                output_path,
                fps=12,
                codec='libx264',
                audio_codec='aac' if audio_clip else None,
                verbose=False,
                logger=None
            )
            
            # Cleanup
            final_clip.close()
            if audio_clip:
                audio_clip.close()
            os.remove(temp_img_path)
            
            print(f"✅ Fallback video created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Fallback video creation failed: {e}")
            return None
    
    def create_simple_audio(self, text, duration):
        """Create simple TTS audio (placeholder - can be enhanced)"""
        try:
            # This is a placeholder - in a real implementation,
            # you would integrate with your TTS system
            # For now, return None to indicate no audio
            return None
        except Exception as e:
            print(f"⚠️ Simple audio creation failed: {e}")
            return None
    
    def create_error_video(self, error_message, output_filename):
        """Create a video showing error message"""
        try:
            # Create error image
            img = Image.new('RGB', (512, 512), color='#e74c3c')
            draw = ImageDraw.Draw(img)
            
            # Add error message
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            error_text = "Video Generation Failed"
            text_bbox = draw.textbbox((0, 0), error_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (512 - text_width) // 2
            draw.text((text_x, 200), error_text, fill='white', font=font)
            
            # Add retry message
            retry_text = "Please try again later"
            try:
                small_font = ImageFont.truetype("arial.ttf", 16)
            except:
                small_font = ImageFont.load_default()
                
            retry_bbox = draw.textbbox((0, 0), retry_text, font=small_font)
            retry_width = retry_bbox[2] - retry_bbox[0]
            retry_x = (512 - retry_width) // 2
            draw.text((retry_x, 250), retry_text, fill='#ecf0f1', font=small_font)
            
            # Save and create video
            temp_img_path = tempfile.mktemp(suffix='.png')
            img.save(temp_img_path)
            
            img_clip = ImageClip(temp_img_path, duration=10)
            output_path = os.path.join(self.output_dir, output_filename)
            
            img_clip.write_videofile(
                output_path,
                fps=12,
                codec='libx264',
                verbose=False,
                logger=None
            )
            
            img_clip.close()
            os.remove(temp_img_path)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error video creation failed: {e}")
            return None

# Global fallback generator
fallback_generator = FallbackGenerator()
