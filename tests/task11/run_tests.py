#!/usr/bin/env python3
"""
Test Runner for Adaptive Engine Day 1 Tests
Runs tests with proper path setup
"""
import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
animatediff_path = project_root / "AnimateDiff"

# Add to path and change directory
sys.path.insert(0, str(animatediff_path))
os.chdir(str(animatediff_path))

print(f"✅ Project root: {project_root}")
print(f"✅ AnimateDiff path: {animatediff_path}")
print(f"✅ Working directory: {os.getcwd()}")
print()

# Now run pytest
import pytest

# Run tests
exit_code = pytest.main([
    str(project_root / "tests" / "adaptive_engine" / "test_story_context_parser.py"),
    str(project_root / "tests" / "adaptive_engine" / "test_identity_memory.py"),
    "-v",
    "--tb=short",
    "-x"
])

sys.exit(exit_code)
