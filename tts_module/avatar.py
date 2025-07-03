import streamlit as st
import requests
import os
import time

# ========================
# Streamlit Configuration
# ========================
st.set_page_config(page_title="LipSync Avatar Generator", layout="centered")

st.title("🎙️ LipSync Avatar Generator")
st.markdown("Enter text below and generate a lip-synced video using AI avatars.")

# ========================
# Backend Configuration
# ========================
API_URL = "http://192.168.0.125:8001/api/generate-and-sync"  # Update IP if needed
TIMEOUT = 300  # seconds

# ========================
# Video Display Helper
# ========================
def display_video(video_path):
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        st.video(video_bytes)
        return video_bytes

# ========================
# Main App
# ========================
def main():
    text_input = st.text_area("Enter text to synthesize", height=150, max_chars=3000)

    lang_code = st.selectbox(
        "Choose target language",
        options=[
            "en", "hi", "mr", "ta", "te", "kn", "ml", "gu", "bn", "pa",
            "es", "fr", "de", "zh", "ja", "ru", "ar", "pt", "it"
        ],
        format_func=lambda code: f"{code.upper()} - {code}"
    )

    if st.button("Generate Lip-Synced Video"):
        if not text_input.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Processing... This might take a minute or two."):
            try:
                response = requests.post(
                    API_URL,
                    data={"text": text_input.strip(), "target_lang": lang_code},
                    timeout=TIMEOUT
                )

                if response.status_code == 200:
                    video_filename = f"result_{int(time.time())}.mp4"
                    with open(video_filename, "wb") as f:
                        f.write(response.content)

                    st.success("Video generated successfully!")
                    video_bytes = display_video(video_filename)

                    if video_bytes:
                        st.download_button(
                            label="Download Video",
                            data=video_bytes,
                            file_name=video_filename,
                            mime="video/mp4"
                        )
                else:
                    try:
                        error_msg = response.json().get("detail", "Unknown error")
                    except:
                        error_msg = response.text
                    st.error(f"Server error: {error_msg}")

            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    main()
