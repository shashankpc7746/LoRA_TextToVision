"""
Audio Manager for Task-7 Quality Leap
Enhanced SadTalker/VASA-1 with micro-expressions
"""

from .enhanced_sadtalker import EnhancedSadTalker
from .vasa_integrator import VASAIntegrator
from .micro_expression import MicroExpressionEngine
from .audio_pipeline import AudioPipeline

__all__ = [
    'EnhancedSadTalker',
    'VASAIntegrator',
    'MicroExpressionEngine',
    'AudioPipeline'
]