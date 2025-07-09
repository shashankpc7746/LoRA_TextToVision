#!/usr/bin/env python3
"""
Simple script to start the AnimateDiff API server
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting AnimateDiff API Server...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found!")
        print("Please run this script from the AnimateDiff_API directory")
        return
    
    print("✅ Found main.py")
    print("🌐 Starting FastAPI server...")
    print("📱 The API will be available at http://localhost:8000")
    print("📖 API docs will be available at http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        # Run uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()
