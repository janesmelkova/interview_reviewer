import base64
import streamlit as st
import whisper
import tempfile
import torch
import asyncio
import os
from dotenv import load_dotenv
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
import threading
from moviepy.editor import VideoFileClip

load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
MODEL_NAME = "mistral-large-latest"

if not MISTRAL_API_KEY:
    st.error("MISTRAL API Key is missing. Please set it in your environment.")
    st.stop()

def clear_gpu_memory():
    """Clears GPU memory."""
    torch.cuda.empty_cache()

@st.cache_resource
def load_model():
    """Loads the Whisper model and returns it with the device."""
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large-v3-turbo").to(device)
    return model, device

def extract_audio_from_video(video_data):
    """Extracts audio from video file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_data)
        temp_video.flush()
        video_clip = VideoFileClip(temp_video.name)
        audio_data = video_clip.audio
        audio_file_path = temp_video.name.replace(".mp4", ".wav")
        audio_data.write_audiofile(audio_file_path, codec="pcm_s16le")
        return audio_file_path

def transcribe_audio(model, device, audio_data, language, status_text):
    """
    Transcribes audio data using the provided model and device.

    Parameters:
    - model: The Whisper model to use for transcription.
    - device: The device (CPU/GPU) to use for transcription.
    - audio_data: The audio data to transcribe.
    - language: The language of the audio data.
    - status_text: Streamlit status text element to update the status.

    Returns:
    - The transcribed text or an error message.
    """
    status_text.text("Transcription in progress...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_data)
        temp_audio.flush()
        try:
            result = model.transcribe(temp_audio.name, language=language, fp16=False)
            transcription_text = result["text"]
        except RuntimeError as e:
            status_text.text(f"Transcription failed: {e}")
            return f"Transcription error: {e}"
        finally:
            try:
                os.remove(temp_audio.name)
            except OSError as e:
                st.error(f"Failed to remove temp file: {e}")
    status_text.text("Transcription complete")
    return transcription_text

async def evaluate_translation(original_text, translated_text, source_language, target_language, status_text):
    """
    Evaluates the translation using the Mistral API.

    Parameters:
    - original_text: The original transcribed text.
    - translated_text: The translated transcribed text.
    - source_language: The language of the original text.
    - target_language: The language of the translated text.
    - status_text: Streamlit status text element to update the status.

    Returns:
    - The evaluation result or an error message.
    """
    status_text.text("Evaluation in progress...")
    client = MistralAsyncClient(api_key=MISTRAL_API_KEY)

    prompt = f"""
    You'll be given a transcription of oral interpretation from {source_language} to {target_language}. 
    You need to evaluate the quality of interpretation considering both content and formality. 
    While evaluating, keep in mind that it's a transcript of oral interpretation. 
    Provide:
    1) scores for the following criteria, where each score is between 0 and 10:

    Content (weight 1):
    1.1. Meaning
    1.2. Completeness
    1.3. Correspondence of numbers with consideration of rounding

    Formality (weight 1):
    2.1. Authenticity 
    2.2. Syntax 
    2.3. Morphology
    2.4. Correspondence of style
    

    2) examples of mistakes for each point.
    
    3) the total score on a ten-point scale (as the average of all marks above)


    Original text:
    {original_text}

    Translated text:
    {translated_text}
    """

    messages = [ChatMessage(role="user", content=prompt)]
    async_response = client.chat_stream(model=MODEL_NAME, messages=messages)

    evaluation_result = ""
    async for chunk in async_response:
        evaluation_result += chunk.choices[0].delta.content

    status_text.text("Evaluation complete")
    return evaluation_result

def evaluate_translation_sync(*args):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(evaluate_translation(*args))

def display_logo(logo_path):
    """Displays the logo if the file exists."""
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as logo_file:
            logo_data = base64.b64encode(logo_file.read()).decode()
        logo_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_data}" style="max-width: 600px; max-height: 600px; display: block; margin: auto;" /></div>'
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        st.warning("Logo file not found")

def main():
    """Main function to run the Streamlit app."""
    model, device = load_model()

    display_logo("ib_logo 1.jpg")

    header_html = '<p style="color:#1a7fe3; font-size: 24px; text-align: center;">Interview Reviewer</p>'
    st.markdown(header_html, unsafe_allow_html=True)

    languages = {
        "English": "en",
        "Russian": "ru",
        "Uzbek": "uz",
        "Chinese": "zh"
    }

    selected_original_language = st.selectbox("Select original language", list(languages.keys()))
    selected_translation_language = st.selectbox("Select translation language", list(languages.keys()))

    original_language = languages[selected_original_language]
    translation_language = languages[selected_translation_language]

    original_file = st.file_uploader("Upload original audio/video file", type=["mp3", "mp4"])
    translated_file = st.file_uploader("Upload translated audio/video file", type=["mp3", "mp4"])

    start_analyzing = st.button("Start Analyzing")

    if original_file is not None and translated_file is not None and start_analyzing:
        transcription_status = st.empty()
        evaluation_status = st.empty()

        original_file_data = original_file.read()
        translated_file_data = translated_file.read()

        # Check if it's a video or audio
        if original_file.type == "video/mp4":
            original_audio_path = extract_audio_from_video(original_file_data)
            with open(original_audio_path, 'rb') as audio_file:
                original_audio_data = audio_file.read()
        else:
            original_audio_data = original_file_data

        if translated_file.type == "video/mp4":
            translated_audio_path = extract_audio_from_video(translated_file_data)
            with open(translated_audio_path, 'rb') as audio_file:
                translated_audio_data = audio_file.read()
        else:
            translated_audio_data = translated_file_data

        # Transcribe and evaluate with status updates
        original_transcription_text = transcribe_audio(model, device, original_audio_data, original_language, transcription_status)
        translation_transcription_text = transcribe_audio(model, device, translated_audio_data, translation_language, transcription_status)

        if "Transcription error" in original_transcription_text or "Transcription error" in translation_transcription_text:
            st.error(f"Transcription failed: {original_transcription_text} {translation_transcription_text}")
            return

        st.text_area("Original Transcription", original_transcription_text)
        st.text_area("Translated Transcription", translation_transcription_text)

        # Running the async function synchronously using threading
        evaluation_thread = threading.Thread(target=evaluate_translation_sync, args=(original_transcription_text, translation_transcription_text, original_language, translation_language, evaluation_status))
        evaluation_thread.start()
        evaluation_thread.join()

        evaluation = evaluate_translation_sync(original_transcription_text, translation_transcription_text, original_language, translation_language, evaluation_status)

        if "Evaluation error" in evaluation:
            st.error(f"Evaluation failed: {evaluation}")
            return

        st.markdown("### Translation evaluation by MISTRAL:")
        st.text_area("Evaluation", evaluation)

        st.markdown("### Download evaluation as a text file")
        st.download_button(
            label="Download Evaluation",
            data=evaluation.encode("utf-8"),
            file_name="evaluation.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()