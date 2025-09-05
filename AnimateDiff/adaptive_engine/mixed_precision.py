"""
Mixed Precision Manager for Task 4 Day 3
Automatic mixed precision configuration for optimal performance
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class PrecisionMode(Enum):
    """Precision modes"""
    FP32 = "fp32"      # Full precision
    FP16 = "fp16"      # Half precision
    BF16 = "bf16"      # BFloat16
    INT8 = "int8"      # 8-bit quantization
    AUTO = "auto"      # Automatic selection


class DeviceType(Enum):
    """Device types"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"        # Apple Silicon
    ROCM = "rocm"      # AMD GPUs


@dataclass
class PrecisionConfig:
    """Mixed precision configuration"""
    mode: PrecisionMode
    device_type: DeviceType
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    scaler_enabled: bool = False
    memory_efficient: bool = True

    # Model-specific settings
    attention_precision: PrecisionMode = PrecisionMode.FP16
    vae_precision: PrecisionMode = PrecisionMode.FP32
    text_encoder_precision: PrecisionMode = PrecisionMode.FP16

    # Performance settings
    compile_model: bool = False
    use_flash_attention: bool = True
    use_memory_efficient_attention: bool = True


class MixedPrecisionManager:
    """Manages mixed precision settings for optimal performance"""

    def __init__(self):
        self.device_capabilities = self._detect_device_capabilities()
        self.precision_configs = self._create_default_configs()

    def _detect_device_capabilities(self) -> Dict[str, Any]:
        """Detect device capabilities for precision optimization"""
        capabilities = {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": hasattr(torch, 'mps') and torch.mps.is_available(),
            "cpu_count": torch.get_num_threads(),
            "cuda_version": None,
            "gpu_memory_gb": 0,
            "gpu_name": None,
            "supports_bfloat16": False,
            "supports_float16": False,
            "supports_int8": False
        }

        if capabilities["cuda_available"]:
            capabilities["cuda_version"] = torch.version.cuda
            capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            capabilities["gpu_name"] = torch.cuda.get_device_name(0)
            capabilities["supports_bfloat16"] = torch.cuda.is_bf16_supported()
            capabilities["supports_float16"] = True
            capabilities["supports_int8"] = True

        if capabilities["mps_available"]:
            capabilities["supports_float16"] = True
            capabilities["supports_bfloat16"] = True

        return capabilities

    def _create_default_configs(self) -> Dict[str, PrecisionConfig]:
        """Create default precision configurations for different scenarios"""
        configs = {}

        # RTX 30-series and above (high-end)
        if (self.device_capabilities["cuda_available"] and
            self.device_capabilities["gpu_memory_gb"] >= 8):

            configs["high_end_cuda"] = PrecisionConfig(
                mode=PrecisionMode.FP16,
                device_type=DeviceType.CUDA,
                scaler_enabled=True,
                compile_model=True,
                use_flash_attention=True
            )

            configs["high_end_cuda_memory"] = PrecisionConfig(
                mode=PrecisionMode.BF16,
                device_type=DeviceType.CUDA,
                scaler_enabled=True,
                memory_efficient=True,
                use_flash_attention=True
            )

        # RTX 20-series and below (mid-range)
        elif (self.device_capabilities["cuda_available"] and
              self.device_capabilities["gpu_memory_gb"] >= 4):

            configs["mid_range_cuda"] = PrecisionConfig(
                mode=PrecisionMode.FP16,
                device_type=DeviceType.CUDA,
                scaler_enabled=True,
                gradient_accumulation_steps=2,
                use_flash_attention=False
            )

        # Apple Silicon
        elif self.device_capabilities["mps_available"]:
            configs["apple_silicon"] = PrecisionConfig(
                mode=PrecisionMode.BF16,
                device_type=DeviceType.MPS,
                scaler_enabled=True,
                memory_efficient=True
            )

        # CPU fallback
        configs["cpu"] = PrecisionConfig(
            mode=PrecisionMode.FP32,
            device_type=DeviceType.CPU,
            memory_efficient=True,
            use_flash_attention=False
        )

        # Office GPU pool (simulated high-end)
        configs["office_gpu"] = PrecisionConfig(
            mode=PrecisionMode.FP16,
            device_type=DeviceType.CUDA,
            scaler_enabled=True,
            compile_model=True,
            use_flash_attention=True,
            gradient_accumulation_steps=1
        )

        return configs

    def get_optimal_config(self, device_class: str = "auto",
                          memory_pressure: str = "normal",
                          task_complexity: str = "medium") -> PrecisionConfig:
        """
        Get optimal precision configuration for given conditions

        Args:
            device_class: Device class (auto, local, office_gpu, cloud)
            memory_pressure: Memory pressure (low, normal, high)
            task_complexity: Task complexity (simple, medium, complex)

        Returns:
            Optimal PrecisionConfig
        """
        # Auto-detect device class
        if device_class == "auto":
            if self.device_capabilities["cuda_available"]:
                if self.device_capabilities["gpu_memory_gb"] >= 8:
                    device_class = "high_end_cuda"
                else:
                    device_class = "mid_range_cuda"
            elif self.device_capabilities["mps_available"]:
                device_class = "apple_silicon"
            else:
                device_class = "cpu"

        # Adjust for memory pressure
        if memory_pressure == "high":
            # Use more aggressive precision for high memory pressure
            if device_class in ["high_end_cuda", "office_gpu"]:
                device_class = f"{device_class}_memory"

        # Adjust for task complexity
        if task_complexity == "simple":
            # Simple tasks can use higher precision for better quality
            pass  # Keep default
        elif task_complexity == "complex":
            # Complex tasks benefit from mixed precision
            pass  # Keep optimized settings

        # Get configuration
        if device_class in self.precision_configs:
            config = self.precision_configs[device_class]
        else:
            # Fallback to CPU config
            config = self.precision_configs["cpu"]

        return config

    def apply_precision_config(self, model, config: PrecisionConfig):
        """
        Apply precision configuration to a model

        Args:
            model: PyTorch model
            config: PrecisionConfig to apply

        Returns:
            Configured model and any additional objects (scaler, etc.)
        """
        # Move model to appropriate device
        device = self._get_device(config.device_type)
        model = model.to(device)

        # Apply precision settings
        if config.mode == PrecisionMode.FP16 and config.scaler_enabled:
            model = model.half()
            scaler = torch.cuda.amp.GradScaler()
        elif config.mode == PrecisionMode.BF16:
            model = model.bfloat16()
            scaler = None
        else:
            scaler = None

        # Apply memory optimizations
        if config.memory_efficient:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()

        return model, scaler, device

    def _get_device(self, device_type: DeviceType):
        """Get PyTorch device object"""
        if device_type == DeviceType.CUDA and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_type == DeviceType.MPS and hasattr(torch, 'mps') and torch.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def get_memory_optimization_tips(self, config: PrecisionConfig) -> List[str]:
        """Get memory optimization tips for the configuration"""
        tips = []

        if config.mode in [PrecisionMode.FP16, PrecisionMode.BF16]:
            tips.append("Using mixed precision - reduces memory usage by ~50%")

        if config.gradient_accumulation_steps > 1:
            tips.append(f"Gradient accumulation ({config.gradient_accumulation_steps} steps) - effective batch size increased")

        if config.memory_efficient:
            tips.append("Memory efficient attention enabled")

        if config.use_flash_attention:
            tips.append("Flash attention enabled for faster processing")

        if config.compile_model:
            tips.append("Model compilation enabled for better performance")

        return tips

    def estimate_memory_usage(self, config: PrecisionConfig,
                            model_params: int, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage for given configuration

        Args:
            config: Precision configuration
            model_params: Number of model parameters
            batch_size: Batch size

        Returns:
            Memory usage estimates in GB
        """
        # Base memory per parameter (rough estimates)
        bytes_per_param = {
            PrecisionMode.FP32: 4,
            PrecisionMode.FP16: 2,
            PrecisionMode.BF16: 2,
            PrecisionMode.INT8: 1
        }

        param_memory = model_params * bytes_per_param.get(config.mode, 4) / (1024**3)  # GB

        # Activation memory (rough estimate)
        activation_memory = param_memory * 2 * batch_size

        # Gradient memory
        gradient_memory = param_memory * batch_size

        # Optimizer memory (Adam)
        optimizer_memory = param_memory * 2 * batch_size

        total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory

        return {
            "model_params": param_memory,
            "activations": activation_memory,
            "gradients": gradient_memory,
            "optimizer": optimizer_memory,
            "total": total_memory,
            "efficiency_ratio": 4 / bytes_per_param.get(config.mode, 4)  # vs FP32
        }

    def get_precision_stats(self) -> Dict[str, Any]:
        """Get precision system statistics"""
        return {
            "device_capabilities": self.device_capabilities,
            "available_configs": list(self.precision_configs.keys()),
            "recommended_config": self.get_optimal_config().__dict__,
            "memory_efficient": all(config.memory_efficient for config in self.precision_configs.values())
        }


# Global mixed precision instance
_mixed_precision = None

def get_mixed_precision() -> MixedPrecisionManager:
    """Get global mixed precision instance"""
    global _mixed_precision
    if _mixed_precision is None:
        _mixed_precision = MixedPrecisionManager()
    return _mixed_precision