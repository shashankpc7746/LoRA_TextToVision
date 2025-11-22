"""
Telemetry v3 - Day 6 of TTV Studio Intelligence Stack

Purpose:
    Extended telemetry for TTV Studio with 20+ new metrics.
    Completes Phase 2 Goal #2: Dashboard Backend (final 25%)

Features:
    1. Story Analysis Metrics - Character detection, gender resolution, text condensation
    2. Scene Graph Metrics - Entity tracking, scene transitions
    3. Narrative Metrics - Story beats, character arcs, tension curves
    4. Emotion Metrics - Emotion changes, motion intensity
    5. Extension Metrics - Video extension stats, quality preservation
    6. Performance Metrics - Module execution times, cache hits

Exports:
    - JSON format for web dashboards
    - Prometheus format for monitoring systems
    - Real-time metric streaming

Author: TTV Studio Team
Created: November 22, 2025
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from enum import Enum


class MetricCategory(Enum):
    """Metric categories"""
    STORY_ANALYSIS = "story_analysis"
    SCENE_GRAPH = "scene_graph"
    NARRATIVE = "narrative"
    EMOTION = "emotion"
    EXTENSION = "extension"
    PERFORMANCE = "performance"
    QUALITY = "quality"


@dataclass
class MetricSnapshot:
    """Single metric snapshot"""
    timestamp: float
    category: str
    name: str
    value: Any
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetrySession:
    """Single telemetry session for one video generation"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    lesson_name: str = ""
    style: str = "realistic"
    metrics: List[MetricSnapshot] = field(default_factory=list)
    
    def duration(self) -> float:
        """Get session duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class TelemetryV3:
    """
    Telemetry v3 - Extended metrics for TTV Studio
    
    Tracks 20+ metrics across 7 categories:
    - Story Analysis (character detection, text condensation)
    - Scene Graph (entities, transitions)
    - Narrative (beats, arcs, tension)
    - Emotion (changes, motion intensity)
    - Extension (clips extended, methods used)
    - Performance (execution times, cache)
    - Quality (frame quality, sync accuracy)
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(TelemetryV3, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize telemetry"""
        if self._initialized:
            return
        
        self.sessions: Dict[str, TelemetrySession] = {}
        self.current_session: Optional[TelemetrySession] = None
        self.export_dir = Path("analytics/telemetry")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self._initialized = True
        print("📊 Telemetry v3 initialized - TTV Studio Metrics")
    
    # ======================== SESSION MANAGEMENT ========================
    
    def start_session(self, lesson_name: str = "", style: str = "realistic") -> str:
        """Start new telemetry session"""
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.current_session = TelemetrySession(
            session_id=session_id,
            start_time=time.time(),
            lesson_name=lesson_name,
            style=style
        )
        
        self.sessions[session_id] = self.current_session
        print(f"📊 Telemetry session started: {session_id}")
        return session_id
    
    def end_session(self) -> Optional[str]:
        """End current telemetry session"""
        if not self.current_session:
            return None
        
        self.current_session.end_time = time.time()
        duration = self.current_session.duration()
        session_id = self.current_session.session_id
        
        print(f"📊 Telemetry session ended: {session_id} ({duration:.1f}s)")
        
        # Auto-export session
        self.export_session_json(session_id)
        
        self.current_session = None
        return session_id
    
    # ======================== METRIC RECORDING ========================
    
    def record_metric(
        self,
        category: MetricCategory,
        name: str,
        value: Any,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a metric in current session"""
        if not self.current_session:
            return
        
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            category=category.value,
            name=name,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )
        
        self.current_session.metrics.append(snapshot)
    
    # ======================== STORY ANALYSIS METRICS ========================
    
    def record_story_analysis(
        self,
        character_count: int,
        gender_resolved: bool,
        condensation_percent: float,
        original_length: int,
        condensed_length: int
    ):
        """Record story analysis metrics"""
        self.record_metric(
            MetricCategory.STORY_ANALYSIS,
            "character_count",
            character_count,
            "characters"
        )
        
        self.record_metric(
            MetricCategory.STORY_ANALYSIS,
            "gender_resolved",
            gender_resolved,
            "boolean"
        )
        
        self.record_metric(
            MetricCategory.STORY_ANALYSIS,
            "text_condensation_percent",
            condensation_percent,
            "%",
            {"original": original_length, "condensed": condensed_length}
        )
    
    # ======================== SCENE GRAPH METRICS ========================
    
    def record_scene_graph(
        self,
        total_scenes: int,
        total_entities: int,
        avg_entities_per_scene: float,
        transitions_detected: int
    ):
        """Record scene graph metrics"""
        self.record_metric(
            MetricCategory.SCENE_GRAPH,
            "total_scenes",
            total_scenes,
            "scenes"
        )
        
        self.record_metric(
            MetricCategory.SCENE_GRAPH,
            "total_entities",
            total_entities,
            "entities"
        )
        
        self.record_metric(
            MetricCategory.SCENE_GRAPH,
            "avg_entities_per_scene",
            avg_entities_per_scene,
            "entities/scene"
        )
        
        self.record_metric(
            MetricCategory.SCENE_GRAPH,
            "transitions_detected",
            transitions_detected,
            "transitions"
        )
    
    # ======================== NARRATIVE METRICS ========================
    
    def record_narrative(
        self,
        story_beats: int,
        character_arcs: int,
        avg_tension: float,
        peak_tension: float,
        pacing_distribution: Dict[str, int]
    ):
        """Record narrative analysis metrics"""
        self.record_metric(
            MetricCategory.NARRATIVE,
            "story_beats",
            story_beats,
            "beats"
        )
        
        self.record_metric(
            MetricCategory.NARRATIVE,
            "character_arcs",
            character_arcs,
            "arcs"
        )
        
        self.record_metric(
            MetricCategory.NARRATIVE,
            "avg_tension",
            avg_tension,
            "0-1 scale"
        )
        
        self.record_metric(
            MetricCategory.NARRATIVE,
            "peak_tension",
            peak_tension,
            "0-1 scale"
        )
        
        self.record_metric(
            MetricCategory.NARRATIVE,
            "pacing_distribution",
            pacing_distribution,
            "scenes",
            {"breakdown": pacing_distribution}
        )
    
    # ======================== EMOTION METRICS ========================
    
    def record_emotion(
        self,
        emotion_changes: int,
        avg_motion_intensity: float,
        emotion_distribution: Dict[str, int]
    ):
        """Record emotion controller metrics"""
        self.record_metric(
            MetricCategory.EMOTION,
            "emotion_changes",
            emotion_changes,
            "changes"
        )
        
        self.record_metric(
            MetricCategory.EMOTION,
            "avg_motion_intensity",
            avg_motion_intensity,
            "multiplier"
        )
        
        self.record_metric(
            MetricCategory.EMOTION,
            "emotion_distribution",
            emotion_distribution,
            "emotions",
            {"breakdown": emotion_distribution}
        )
    
    # ======================== EXTENSION METRICS ========================
    
    def record_extension(
        self,
        clips_extended: int,
        total_clips: int,
        slowmo_count: int,
        freeze_count: int,
        avg_extension_duration: float,
        quality_preserved: bool
    ):
        """Record video extension metrics"""
        self.record_metric(
            MetricCategory.EXTENSION,
            "clips_extended",
            clips_extended,
            "clips",
            {"total_clips": total_clips, "extension_rate": clips_extended/total_clips if total_clips > 0 else 0}
        )
        
        self.record_metric(
            MetricCategory.EXTENSION,
            "slowmo_count",
            slowmo_count,
            "clips"
        )
        
        self.record_metric(
            MetricCategory.EXTENSION,
            "freeze_count",
            freeze_count,
            "clips"
        )
        
        self.record_metric(
            MetricCategory.EXTENSION,
            "avg_extension_duration",
            avg_extension_duration,
            "seconds"
        )
        
        self.record_metric(
            MetricCategory.EXTENSION,
            "quality_preserved",
            quality_preserved,
            "boolean"
        )
    
    # ======================== PERFORMANCE METRICS ========================
    
    def record_performance(
        self,
        module_name: str,
        execution_time: float,
        cache_hits: int = 0,
        cache_misses: int = 0
    ):
        """Record module performance metrics"""
        self.record_metric(
            MetricCategory.PERFORMANCE,
            f"{module_name}_execution_time",
            execution_time,
            "seconds",
            {"module": module_name}
        )
        
        if cache_hits > 0 or cache_misses > 0:
            total = cache_hits + cache_misses
            hit_rate = cache_hits / total if total > 0 else 0
            
            self.record_metric(
                MetricCategory.PERFORMANCE,
                f"{module_name}_cache_hit_rate",
                hit_rate,
                "%",
                {"hits": cache_hits, "misses": cache_misses}
            )
    
    # ======================== QUALITY METRICS ========================
    
    def record_quality(
        self,
        audio_video_sync_diff: float,
        avg_frame_quality: float,
        bitrate: int,
        fps: float
    ):
        """Record output quality metrics"""
        self.record_metric(
            MetricCategory.QUALITY,
            "audio_video_sync_diff",
            audio_video_sync_diff,
            "seconds"
        )
        
        self.record_metric(
            MetricCategory.QUALITY,
            "avg_frame_quality",
            avg_frame_quality,
            "0-1 scale"
        )
        
        self.record_metric(
            MetricCategory.QUALITY,
            "output_bitrate",
            bitrate,
            "kbps"
        )
        
        self.record_metric(
            MetricCategory.QUALITY,
            "output_fps",
            fps,
            "fps"
        )
    
    # ======================== EXPORT FUNCTIONS ========================
    
    def export_session_json(self, session_id: str) -> Optional[Path]:
        """Export session to JSON file"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        output_file = self.export_dir / f"{session_id}.json"
        
        # Convert session to dict
        session_dict = {
            "session_id": session.session_id,
            "start_time": datetime.fromtimestamp(session.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(session.end_time).isoformat() if session.end_time else None,
            "duration_seconds": session.duration(),
            "lesson_name": session.lesson_name,
            "style": session.style,
            "metrics": [
                {
                    "timestamp": datetime.fromtimestamp(m.timestamp).isoformat(),
                    "category": m.category,
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "metadata": m.metadata
                }
                for m in session.metrics
            ]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Exported telemetry: {output_file}")
        return output_file
    
    def export_prometheus(self, session_id: str) -> Optional[str]:
        """Export session metrics in Prometheus format"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        lines = []
        
        # Group metrics by category and name
        for metric in session.metrics:
            metric_name = f"ttv_studio_{metric.category}_{metric.name}".replace(" ", "_").lower()
            
            # Convert value to number if possible
            try:
                value = float(metric.value) if isinstance(metric.value, (int, float)) else 1
            except:
                value = 1  # Boolean or string metrics default to 1
            
            # Add labels
            labels = {
                "lesson": session.lesson_name,
                "style": session.style,
                "session": session.session_id[:8]
            }
            labels.update(metric.metadata)
            
            label_str = ",".join([f'{k}="{v}"' for k, v in labels.items()])
            lines.append(f"{metric_name}{{{label_str}}} {value}")
        
        prometheus_output = "\n".join(lines)
        
        # Save to file
        output_file = self.export_dir / f"{session_id}.prom"
        with open(output_file, 'w') as f:
            f.write(prometheus_output)
        
        print(f"📊 Exported Prometheus metrics: {output_file}")
        return prometheus_output
    
    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session summary statistics"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Aggregate metrics by category
        summary = {
            "session_id": session_id,
            "duration": session.duration(),
            "lesson_name": session.lesson_name,
            "total_metrics": len(session.metrics),
            "categories": {}
        }
        
        for category in MetricCategory:
            cat_metrics = [m for m in session.metrics if m.category == category.value]
            summary["categories"][category.value] = {
                "count": len(cat_metrics),
                "metrics": [m.name for m in cat_metrics]
            }
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall telemetry statistics"""
        return {
            "total_sessions": len(self.sessions),
            "active_session": self.current_session.session_id if self.current_session else None,
            "export_dir": str(self.export_dir),
            "categories": [c.value for c in MetricCategory]
        }


# ======================== SINGLETON ACCESSOR ========================

_telemetry_instance = None

def get_telemetry() -> TelemetryV3:
    """Get singleton telemetry instance"""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryV3()
    return _telemetry_instance
