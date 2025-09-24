"""
Frame Interpolator for Task-7 Quality Leap
RIFE wrapper with caching for smooth 24-30fps video
"""

from .rife_interpolator import RIFEInterpolator
from .frame_cache import FrameCache
from .interpolation_pipeline import InterpolationPipeline

__all__ = [
    'RIFEInterpolator',
    'FrameCache',
    'InterpolationPipeline'
]