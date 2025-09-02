#!/usr/bin/env python3
"""
Device Probe Module - Task-4 Day-1
Detects local hardware capabilities and determines optimal processing tier
"""

import torch
import psutil
import platform
import subprocess
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
try:
    import GPUtil  # type: ignore
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None  # type: ignore
    GPUTIL_AVAILABLE = False

try:
    import cpuinfo  # type: ignore
    CPUINFO_AVAILABLE = True
except ImportError:
    cpuinfo = None  # type: ignore
    CPUINFO_AVAILABLE = False


@dataclass
class DeviceCapabilities:
    """Device capability assessment"""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_utilization: float = 0.0
    cuda_version: str = ""
    driver_version: str = ""
    cpu_cores: int = 0
    cpu_name: str = ""
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    has_webgpu: bool = False
    thermal_status: str = "normal"
    battery_level: Optional[float] = None
    can_handle_heavy_load: bool = False
    recommended_tier: str = "local"


class DeviceProbe:
    """Comprehensive device capability detection"""

    def __init__(self):
        self.capabilities = DeviceCapabilities()
        self._probe_completed = False

    def probe_all(self) -> DeviceCapabilities:
        """Complete device capability assessment"""
        if self._probe_completed:
            return self.capabilities

        print("[INFO] Probing device capabilities...")

        # GPU Detection
        self._probe_gpu()

        # CPU Detection
        self._probe_cpu()

        # Memory Detection
        self._probe_memory()

        # System Status
        self._probe_system_status()

        # Capability Assessment
        self._assess_capabilities()

        self._probe_completed = True
        return self.capabilities

    def _probe_gpu(self):
        """Detect GPU capabilities using multiple methods"""
        try:
            # Method 1: PyTorch CUDA
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    # Get primary GPU info
                    self.capabilities.gpu_name = torch.cuda.get_device_name(0)
                    self.capabilities.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                    # Try to get CUDA version
                    try:
                        self.capabilities.cuda_version = torch.version.cuda
                    except:
                        self.capabilities.cuda_version = "Unknown"

            # Method 2: GPUtil for more detailed info (if available)
            if GPUTIL_AVAILABLE and GPUtil is not None:
                try:
                    gpus = GPUtil.getGPUs()  # type: ignore
                    if gpus:
                        gpu = gpus[0]  # Primary GPU
                        self.capabilities.gpu_utilization = gpu.load * 100
                        self.capabilities.driver_version = getattr(gpu, 'driver', 'Unknown')
                except:
                    pass

            # Method 3: nvidia-smi for comprehensive info
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines:
                        parts = [part.strip() for part in lines[0].split(',')]
                        if len(parts) >= 3:
                            self.capabilities.gpu_name = parts[0]
                            # Memory is already in MB from nvidia-smi, convert to GB
                            try:
                                mem_mb = float(parts[1])
                                self.capabilities.gpu_memory_gb = mem_mb / 1024
                            except:
                                pass
                            self.capabilities.driver_version = parts[2]

            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        except Exception as e:
            print(f"[WARNING] GPU probe error: {e}")

    def _probe_cpu(self):
        """Detect CPU capabilities"""
        try:
            self.capabilities.cpu_cores = psutil.cpu_count(logical=True)

            # Get CPU name using cpuinfo (if available)
            if CPUINFO_AVAILABLE and cpuinfo is not None:
                try:
                    info = cpuinfo.get_cpu_info()  # type: ignore
                    self.capabilities.cpu_name = info.get('brand_raw', 'Unknown CPU')
                except:
                    # Fallback to platform
                    self.capabilities.cpu_name = platform.processor() or "Unknown CPU"
            else:
                # Fallback to platform
                self.capabilities.cpu_name = platform.processor() or "Unknown CPU"

        except Exception as e:
            print(f"[WARNING] CPU probe error: {e}")

    def _probe_memory(self):
        """Detect system memory"""
        try:
            mem = psutil.virtual_memory()
            self.capabilities.total_ram_gb = mem.total / (1024**3)
            self.capabilities.available_ram_gb = mem.available / (1024**3)
        except Exception as e:
            print(f"[WARNING] Memory probe error: {e}")

    def _probe_system_status(self):
        """Detect system status (thermal, battery)"""
        try:
            # Battery status (if laptop)
            try:
                if hasattr(psutil, 'sensors_battery') and psutil.sensors_battery():
                    battery = psutil.sensors_battery()
                    self.capabilities.battery_level = battery.percent if battery else None
            except:
                self.capabilities.battery_level = None

            # Thermal status (simplified) - skip if not available
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Check CPU temperature
                        for name, entries in temps.items():
                            if entries:
                                temp_c = entries[0].current
                                if temp_c > 80:
                                    self.capabilities.thermal_status = "hot"
                                elif temp_c > 60:
                                    self.capabilities.thermal_status = "warm"
                                else:
                                    self.capabilities.thermal_status = "normal"
                                break
            except:
                # Thermal sensors not available, keep default "normal"
                pass

        except Exception as e:
            print(f"[WARNING] System status probe error: {e}")

    def _assess_capabilities(self):
        """Assess if device can handle heavy workloads"""
        try:
            # RTX 3060 Ti assessment based on user's system
            gpu_memory_gb = self.capabilities.gpu_memory_gb
            gpu_name = self.capabilities.gpu_name.lower()
            cpu_cores = self.capabilities.cpu_cores
            ram_gb = self.capabilities.total_ram_gb

            # RTX 30-series GPUs can handle most AnimateDiff tasks
            if gpu_memory_gb >= 8.0 and 'rtx' in gpu_name:
                self.capabilities.can_handle_heavy_load = True
                self.capabilities.recommended_tier = "local"
            elif gpu_memory_gb >= 6.0:
                self.capabilities.can_handle_heavy_load = True
                self.capabilities.recommended_tier = "local"
            elif gpu_memory_gb >= 4.0:
                # Can handle medium loads
                self.capabilities.can_handle_heavy_load = False
                self.capabilities.recommended_tier = "local_with_limits"
            else:
                # Low-end GPU, use Yotta for heavy tasks
                self.capabilities.can_handle_heavy_load = False
                self.capabilities.recommended_tier = "yotta_preferred"

            # Additional factors
            if self.capabilities.thermal_status == "hot":
                self.capabilities.can_handle_heavy_load = False
                self.capabilities.recommended_tier = "yotta_preferred"

            if self.capabilities.battery_level and self.capabilities.battery_level < 20:
                self.capabilities.can_handle_heavy_load = False
                self.capabilities.recommended_tier = "yotta_preferred"

        except Exception as e:
            print(f"[WARNING] Capability assessment error: {e}")
            # Safe defaults
            self.capabilities.can_handle_heavy_load = False
            self.capabilities.recommended_tier = "yotta_preferred"

    def get_capabilities_dict(self) -> Dict[str, Any]:
        """Return capabilities as dictionary"""
        if not self._probe_completed:
            self.probe_all()

        return {
            "gpu_name": self.capabilities.gpu_name,
            "gpu_memory_gb": round(self.capabilities.gpu_memory_gb, 1),
            "gpu_utilization": round(self.capabilities.gpu_utilization, 1),
            "cuda_version": self.capabilities.cuda_version,
            "driver_version": self.capabilities.driver_version,
            "cpu_cores": self.capabilities.cpu_cores,
            "cpu_name": self.capabilities.cpu_name,
            "total_ram_gb": round(self.capabilities.total_ram_gb, 1),
            "available_ram_gb": round(self.capabilities.available_ram_gb, 1),
            "thermal_status": self.capabilities.thermal_status,
            "battery_level": self.capabilities.battery_level,
            "can_handle_heavy_load": self.capabilities.can_handle_heavy_load,
            "recommended_tier": self.capabilities.recommended_tier,
            "timestamp": int(time.time())
        }

    def can_handle_task(self, estimated_vram_gb: float, estimated_time_sec: int) -> bool:
        """Check if device can handle a specific task"""
        if not self._probe_completed:
            self.probe_all()

        # Check VRAM availability
        available_vram = self.capabilities.gpu_memory_gb * 0.8  # 80% of total VRAM
        if estimated_vram_gb > available_vram:
            return False

        # Check thermal status
        if self.capabilities.thermal_status == "hot":
            return False

        # Check battery (if applicable)
        if self.capabilities.battery_level and self.capabilities.battery_level < 30:
            return False

        return self.capabilities.can_handle_heavy_load


# Global instance for easy access
device_probe = DeviceProbe()


def get_device_capabilities() -> Dict[str, Any]:
    """Convenience function to get device capabilities"""
    return device_probe.get_capabilities_dict()


def can_handle_heavy_workload() -> bool:
    """Quick check if device can handle heavy workloads"""
    return device_probe.probe_all().can_handle_heavy_load


if __name__ == "__main__":
    # Test the device probe
    probe = DeviceProbe()
    capabilities = probe.probe_all()

    print("[INFO] Device Capabilities Detected:")
    print(json.dumps(probe.get_capabilities_dict(), indent=2))

    # Test task assessment
    print("\n[TEST] Task Assessment Tests:")
    print(f"Can handle 2GB VRAM task: {probe.can_handle_task(2.0, 300)}")
    print(f"Can handle 6GB VRAM task: {probe.can_handle_task(6.0, 300)}")
    print(f"Can handle 10GB VRAM task: {probe.can_handle_task(10.0, 300)}")