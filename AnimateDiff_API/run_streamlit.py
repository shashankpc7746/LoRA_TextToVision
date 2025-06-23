"""
Quick launcher for the improved Streamlit UI
Run this to start the enhanced AnimateDiff interface
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Enhanced AnimateDiff Streamlit UI...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the AnimateDiff_API directory")
        return
    
    print("✅ Found streamlit_app.py")
    print("🌐 Starting Streamlit server...")
    print("📱 The UI will open in your browser automatically")
    print("🔗 URL: http://localhost:8501")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")

if __name__ == "__main__":
    main()
