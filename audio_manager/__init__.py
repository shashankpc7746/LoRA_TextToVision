"""
Audio Manager for Task-7 Quality Leap
Enhanced SadTalker/VASA-1 with micro-expressions
"""

from .enhanced_sadtalker import (EnhancedSadTalker, VASAIntegrator, AudioPipeline,
                                 get_enhanced_sadtalker, get_vasa_integrator, get_audio_pipeline)

__all__ = [
    'EnhancedSadTalker',
    'VASAIntegrator',
    'AudioPipeline',
    'get_enhanced_sadtalker',
    'get_vasa_integrator',
    'get_audio_pipeline'
]