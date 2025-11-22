"""
Pytest configuration for adaptive_engine tests
Handles path setup for importing AnimateDiff modules
"""
import sys
import os
from pathlib import Path

# Add AnimateDiff to Python path
animatediff_path = Path(__file__).parent.parent.parent / "AnimateDiff"
animatediff_str = str(animatediff_path.absolute())

if animatediff_str not in sys.path:
    sys.path.insert(0, animatediff_str)

# Change working directory to AnimateDiff for relative imports
os.chdir(animatediff_str)

print(f"✅ Added AnimateDiff to path: {animatediff_str}")
print(f"✅ Changed working directory to: {os.getcwd()}")
