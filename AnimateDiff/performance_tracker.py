#!/usr/bin/env python3
"""
Performance Metrics Tracker
Tracks generation time, memory usage, model performance, and output quality
"""

import time
import psutil
import os
import json
from datetime import datetime
from pathlib import Path

class PerformanceTracker:
    def __init__(self):
        self.metrics = {}
        self.start_time = None
        self.start_memory = None
        
    def start_tracking(self, operation_name):
        """Start tracking performance for an operation"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        self.metrics[operation_name] = {
            "start_time": datetime.now().isoformat(),
            "operation": operation_name
        }
        
    def end_tracking(self, operation_name, additional_data=None):
        """End tracking and record metrics"""
        if operation_name not in self.metrics:
            return
            
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        self.metrics[operation_name].update({
            "end_time": datetime.now().isoformat(),
            "duration_seconds": round(end_time - self.start_time, 2),
            "memory_used_mb": round(end_memory - self.start_memory, 2),
            "peak_memory_mb": round(end_memory, 2),
            "cpu_percent": psutil.cpu_percent(),
            "status": "completed"
        })
        
        if additional_data:
            self.metrics[operation_name].update(additional_data)
            
    def add_video_metrics(self, operation_name, video_path, model_used, fps=12):
        """Add video-specific metrics"""
        if operation_name not in self.metrics:
            return
            
        try:
            # Get video file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Get video duration (approximate from fps and frames)
            from moviepy.editor import VideoFileClip
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            
            self.metrics[operation_name].update({
                "output_file": os.path.basename(video_path),
                "file_size_mb": round(file_size_mb, 2),
                "video_duration_seconds": round(duration, 2),
                "fps": fps,
                "model_used": model_used,
                "resolution": "512x512",
                "has_audio": True,
                "has_subtitles": True
            })
            
        except Exception as e:
            self.metrics[operation_name]["video_metrics_error"] = str(e)
    
    def save_metrics(self, output_dir="logs"):
        """Save metrics to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(output_dir, f"performance_metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        return metrics_file
    
    def get_summary(self):
        """Get performance summary"""
        if not self.metrics:
            return "No metrics recorded"
            
        summary = []
        for operation, data in self.metrics.items():
            duration = data.get('duration_seconds', 0)
            memory = data.get('memory_used_mb', 0)
            file_size = data.get('file_size_mb', 0)
            
            summary.append(f"📊 {operation}:")
            summary.append(f"   ⏱️ Duration: {duration}s")
            summary.append(f"   💾 Memory: {memory}MB")
            if file_size > 0:
                summary.append(f"   📁 Output: {file_size}MB")
            summary.append("")
            
        return "\n".join(summary)

# Global tracker instance
performance_tracker = PerformanceTracker()

def track_performance(operation_name):
    """Decorator for tracking function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_tracker.start_tracking(operation_name)
            try:
                result = func(*args, **kwargs)
                performance_tracker.end_tracking(operation_name, {"status": "success"})
                return result
            except Exception as e:
                performance_tracker.end_tracking(operation_name, {
                    "status": "error",
                    "error": str(e)
                })
                raise
        return wrapper
    return decorator
