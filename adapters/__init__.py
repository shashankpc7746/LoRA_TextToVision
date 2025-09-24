"""
LoRA Adapters for Task-7 Quality Leap
Fine-tuned adapters for SDXL/AnimateDiff models
"""

from .lora_adapter import LoRAAdapter, GurukulLoRA, get_lora_adapter, get_gurukul_lora
from .adapter_trainer import LoRATrainer, create_gurukul_training_data, quick_train_gurukul_adapter
from .adapter_manager import AdapterManager, get_adapter_manager, quick_setup_gurukul_adapter
from .keyframe_generator import KeyframeGenerator, get_keyframe_generator, generate_keyframes
from .animate_diff_bridge import AnimateDiffBridge, get_animate_diff_bridge, create_keyframe_animation

__all__ = [
    'LoRAAdapter',
    'GurukulLoRA',
    'get_lora_adapter',
    'get_gurukul_lora',
    'LoRATrainer',
    'create_gurukul_training_data',
    'quick_train_gurukul_adapter',
    'AdapterManager',
    'get_adapter_manager',
    'quick_setup_gurukul_adapter',
    'KeyframeGenerator',
    'get_keyframe_generator',
    'generate_keyframes',
    'AnimateDiffBridge',
    'get_animate_diff_bridge',
    'create_keyframe_animation'
]