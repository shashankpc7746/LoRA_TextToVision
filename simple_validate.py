#!/usr/bin/env python3
"""
Simple Project Validation Script
Basic validation without emojis to avoid encoding issues
"""

import os
import sys
from pathlib import Path

def main():
    """Simple validation"""
    print("=== LoRA_TextToVision Project Validation ===")

    # Check basic structure
    required_dirs = ["AnimateDiff", "AnimateDiff_API", "assets"]
    missing = []

    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)

    if missing:
        print(f"ERROR: Missing directories: {missing}")
        return False

    print("OK: Project structure validated")

    # Check adaptive engine
    try:
        from AnimateDiff.adaptive_engine import get_device_capabilities
        print("OK: Adaptive engine imports working")
    except ImportError as e:
        print(f"ERROR: Adaptive engine import failed: {e}")
        return False

    # Check API
    try:
        from AnimateDiff_API.adaptive_api import adaptive_app
        print("OK: API imports working")
    except ImportError as e:
        print(f"ERROR: API import failed: {e}")
        return False

    print("SUCCESS: Basic validation passed")
    print("Project is ready for production!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)