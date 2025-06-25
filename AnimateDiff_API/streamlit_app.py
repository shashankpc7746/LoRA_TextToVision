# streamlit_app.py

import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="AnimateDiff Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better styling and visibility
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #ff6b6b;
        --success-color: #51cf66;
        --warning-color: #ffd43b;
        --info-color: #339af0;
        --dark-bg: #2c3e50;
        --light-bg: #ecf0f1;
    }

    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .subtitle {
        text-align: center;
        color: #34495e;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 25px 0 15px 0;
        border-left: 6px solid #ff6b6b;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-weight: bold;
    }

    .section-header h3 {
        color: white !important;
        margin: 0 !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
    }

    .info-panel {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #339af0;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    .tip-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #2196f3;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(33, 150, 243, 0.2);
        color: #1565c0 !important;
    }

    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #4caf50;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(76, 175, 80, 0.2);
    }

    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #ff9800;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(255, 152, 0, 0.2);
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #e9ecef;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }

    .progress-container {
        background: linear-gradient(135deg, #f1f3f4 0%, #e8eaed 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 2px solid #dadce0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .status-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a73e8;
        text-align: center;
        margin: 10px 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎬 AnimateDiff Video Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Create stunning AI-generated videos with advanced motion synthesis</p>', unsafe_allow_html=True)

# Sidebar with presets and tips
with st.sidebar:
    st.markdown("### 🎨 Quick Presets")

    preset_options = {
        "Custom": {},
        "Anime Character": {
            "prompt": "a beautiful anime girl with flowing hair, walking gracefully in a magical garden, masterpiece, detailed",
            "negative_prompt": "blurry, low quality, deformed face, bad anatomy",
            "guidance_scale": 8.0,
            "steps": 20
        },
        "Fantasy Wizard": {
            "prompt": "a young wizard casting magical spells, blue robes, mystical forest, cinematic lighting",
            "negative_prompt": "blurry, distorted, bad anatomy, low quality",
            "guidance_scale": 8.5,
            "steps": 25
        },
        "Realistic Portrait": {
            "prompt": "a person walking in a beautiful landscape, photorealistic, natural lighting, high quality",
            "negative_prompt": "cartoon, anime, low quality, blurry, distorted",
            "guidance_scale": 7.5,
            "steps": 30
        }
    }

    selected_preset = st.selectbox("Choose a preset:", list(preset_options.keys()))

    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Higher steps** = better quality but slower
    - **Guidance 7-9** works best for most cases
    - **32 frames** = ~2.7 seconds at 12 FPS
    - Use **detailed prompts** for better results
    """)

    st.markdown("### ⚙️ Performance")
    st.info("Estimated generation time: 3-5 minutes")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # ---- Enhanced Form ----
    with st.form("generate_form"):
        st.markdown('<div class="section-header"><h3>📝 Prompt Configuration</h3></div>', unsafe_allow_html=True)

        # Apply preset if selected
        preset_data = preset_options.get(selected_preset, {})

        prompt = st.text_area(
            "✨ Main Prompt",
            value=preset_data.get("prompt", "an anime girl riding a dragon, cinematic, 4K"),
            height=100,
            help="Describe what you want to see in the video. Be specific and detailed."
        )

        negative_prompt = st.text_input(
            "🚫 Negative Prompt",
            value=preset_data.get("negative_prompt", "blurry, distorted, ghost eyes, unnatural skin, motion-blur, scene flicker , extra limbs, low quality"),
            help="What you DON'T want to see in the video"
        )

        st.markdown('<div class="section-header"><h3>🎛️ Generation Parameters</h3></div>', unsafe_allow_html=True)

        # Parameters in columns for better layout
        param_col1, param_col2 = st.columns(2)

        with param_col1:
            seed = st.number_input(
                "🎲 Seed",
                value=333,
                help="Random seed for reproducible results. Use -1 for random."
            )

            guidance_scale = st.slider(
                "🎯 Guidance Scale",
                min_value=1.0,
                max_value=20.0,
                value=preset_data.get("guidance_scale", 15.0),
                step=0.5,
                help="How closely to follow the prompt (7-9 recommended)"
            )

        with param_col2:
            steps = st.slider(
                "⚙️ Inference Steps",
                min_value=10,
                max_value=40,
                value=preset_data.get("steps", 25),
                help="More steps = better quality but slower generation"
            )

            num_frames = st.slider(
                "🎞️ Number of Frames",
                min_value=8,
                max_value=32,
                value=32,
                help="Total frames in the video (32 frames ≈ 2.7 seconds)"
            )

        fps = st.slider(
            "⏱️ FPS (Frames Per Second)",
            min_value=4,
            max_value=24,
            value=8,
            help="Playback speed of the video"
        )

        # Calculate estimated duration
        duration = num_frames / fps
        st.info(f"📊 Estimated video duration: **{duration:.1f} seconds**")

        # Generate button with custom styling
        submit = st.form_submit_button(
            "🚀 Generate Video",
            use_container_width=True,
            type="primary"
        )

with col2:
    st.markdown('<div class="section-header"><h3>📊 Generation Info</h3></div>', unsafe_allow_html=True)

    # Status indicators with enhanced styling
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.markdown("**🔧 Current Settings:**")
    st.write(f"• **Resolution:** 512x512")
    st.write(f"• **Model:** AnimateDiff v1.5")
    st.write(f"• **Scheduler:** Euler Discrete")
    st.markdown('</div>', unsafe_allow_html=True)

    # Performance metrics with enhanced styling
    st.markdown('<div class="info-panel">', unsafe_allow_html=True)
    st.markdown("**⚡ Performance Estimate:**")
    estimated_time = max(1.0, (steps * num_frames) / 80)  # More realistic estimation
    st.write(f"• **Est. Time:** {estimated_time:.1f} minutes")
    st.write(f"• **Memory:** ~6-8 GB VRAM")
    st.write(f"• **Complexity:** {'High' if steps > 25 else 'Medium' if steps > 15 else 'Low'}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced tips box
    st.markdown('<div class="tip-box">', unsafe_allow_html=True)
    st.markdown("**💡 Pro Tips:**")
    st.markdown("""
    - 🎨 Start with presets for best results
    - 🎲 Use seed -1 for random generation
    - ⚡ Lower steps for faster testing
    - 🎯 Higher guidance for prompt adherence
    - 🎞️ 32 frames = ~2.7 seconds video
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---- Enhanced API Call with Realistic Progress ----
if submit:
    # Create enhanced progress container
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    st.markdown("### 🎬 Video Generation in Progress")

    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_estimate = st.empty()

    # Handle random seed
    if seed == -1:
        import random
        seed = random.randint(1, 1000000)
        st.info(f"🎲 Using random seed: {seed}")

    # Calculate realistic time estimate
    estimated_total_time = max(60, (steps * num_frames) / 80 * 60)  # In seconds

    status_text.markdown('<div class="status-text">🔄 Preparing generation request...</div>', unsafe_allow_html=True)
    time_estimate.info(f"⏱️ Estimated time: {estimated_total_time/60:.1f} minutes")
    progress_bar.progress(5 )

    import time
    time.sleep(1)  # Brief pause for user to see initial status

    # Prepare the request payload
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": int(seed),
        "guidance_scale": guidance_scale,
        "steps": steps,
        "num_frames": num_frames,
        "fps": fps
    }

    try:
        status_text.markdown('<div class="status-text">🚀 Connecting to AI backend...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)
        time.sleep(0.5)

        # Send request to FastAPI backend
        api_url = "https://6c63-157-119-202-215.ngrok-free.app/generate-video"

        status_text.markdown('<div class="status-text">🧠 AI model loading and initializing...</div>', unsafe_allow_html=True)
        progress_bar.progress(15)

        # Start the actual request
        start_time = time.time()
        headers = {
            "x-api-key": "shashank_ka_vision786"  # match .env value
        }

        response = requests.post(api_url, json=payload, headers=headers, timeout=900)

        # Simulate realistic progress during generation
        for i in range(20, 85, 5):
            elapsed = time.time() - start_time
            if elapsed > 10:  # After 10 seconds, show generation progress
                frame_progress = min(steps, int(elapsed / (estimated_total_time / steps)))
                status_text.markdown(f'<div class="status-text">� Generating frame {frame_progress}/{steps} - Please wait...</div>', unsafe_allow_html=True)
                progress_bar.progress(i)
                time.sleep(2)

        progress_bar.progress(85)
        status_text.markdown('<div class="status-text">🎬 Finalizing video encoding...</div>', unsafe_allow_html=True)

        if response.status_code == 200:
            video_path = "generated_video.mp4"
            with open(video_path, "wb") as f:
                f.write(response.content)

            progress_bar.progress(100)
            status_text.text("✅ Video generated successfully!")

            # Success message with details
            st.balloons()
            st.success("🎉 **Video Generation Complete!**")

            # Display generation details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎞️ Frames", num_frames)
            with col2:
                st.metric("⏱️ Duration", f"{duration:.1f}s")
            with col3:
                st.metric("🎲 Seed", seed)

            # Display the video
            st.markdown("### 🎬 Generated Video")
            st.video(video_path)

            # Video info
            if os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
                st.info(f"📁 File size: {file_size:.1f} MB | 📍 Path: {os.path.basename(video_path)}")

                # Download button with custom styling
                with open(video_path, "rb") as file:
                    import time
                    st.download_button(
                        label="📥 Download Video",
                        data=file.read(),
                        file_name=f"animatediff_{seed}_{int(time.time())}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        type="secondary"
                    )
        else:
            progress_bar.progress(0)
            status_text.text("❌ Generation failed")
            st.error(f"**Error {response.status_code}:** {response.text}")

            # Error troubleshooting
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common Issues:**
                - Backend server not running
                - Insufficient GPU memory
                - Invalid parameters
                - Network timeout

                **Solutions:**
                - Check if FastAPI server is running on port 8000
                - Reduce number of frames or steps
                - Verify prompt format
                """)

    except requests.exceptions.ConnectionError:
        progress_bar.progress(0)
        status_text.text("❌ Connection failed")
        st.error("🔌 **Cannot connect to backend server**")

        with st.expander("🔧 Backend Setup Instructions"):
            st.markdown("""
            **To start the backend server:**
            ```bash
            cd AnimateDiff_API
            python main.py
            ```

            **Check if server is running:**
            - Open http://127.0.0.1:8000/docs in your browser
            - You should see the FastAPI documentation
            """)

    except requests.exceptions.Timeout:
        progress_bar.progress(0)
        status_text.text("⏰ Request timed out")
        st.error("⏰ **Generation timed out** - This can happen with high-quality settings")
        st.info("💡 Try reducing the number of steps or frames for faster generation")

    except Exception as e:
        progress_bar.progress(0)
        status_text.text("❌ Unexpected error")
        st.error(f"❌ **Unexpected error:** {str(e)}")

# ---- Enhanced Footer ----
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🎬 <strong>AnimateDiff Video Generator</strong></p>
    <p>Built with ❤️ using Streamlit, FastAPI, and AnimateDiff</p>
    <p><em>Create stunning AI-generated videos with motion synthesis</em></p>
</div>
""", unsafe_allow_html=True)
