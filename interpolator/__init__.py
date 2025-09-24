"""
Frame Interpolator for Task-7 Quality Leap
RIFE wrapper with caching for smooth 24-30fps video
"""

from .rife_interpolator import RIFEInterpolator, get_rife_interpolator
from .interpolation_pipeline import InterpolationPipeline, StabilizationEngine, get_interpolation_pipeline

__all__ = [
    'RIFEInterpolator',
    'get_rife_interpolator',
    'InterpolationPipeline',
    'StabilizationEngine',
    'get_interpolation_pipeline'
]