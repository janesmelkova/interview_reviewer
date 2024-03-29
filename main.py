import base64
import streamlit as st
import whisper
import tempfile
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# Load the Whisper model
wisp_model = whisper.load_model("tiny")
client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')
)


def transcribe_audio(wisp_model, audio_path, output_file, language):
    # Load the audio file and transcribe it with progress bar and timer
    result = wisp_model.transcribe(
        audio_path,
        language=language,
        fp16=False,
        verbose=True,
    )

    with open(output_file, "w") as f:
        f.write(result["text"])

    print(f"Transcription results saved to {output_file}")
    return output_file

def evaluate_translation(original_text, translated_text, source_language, target_language):
    prompt = f"You'll be given a transcription of oral interpretation from {source_language} to {target_language}. You need to evaluate the quality of interpretation considering both content and formality. Provide scores for the following criteria, where each score is between 0 and 10:\n\nContent (weight 1):\n1.1. Correspondence of names, titles\n1.2. Correspondence of numbers with consideration of rounding\n1.3. Meaning\n\nFormality (weight 1):\n2.1. Authenticity of phrases\n2.2. Syntax (sentence completion)\n2.3. Morphology (agreement of forms)\nProvide examples of mistakes for each point\nOriginal text:\n{original_text}\n\nTranslated text:\n{translated_text}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
        #top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        #n=3,
    )
    evaluation = response.choices[0].message.content
    return evaluation


def main():
    # Set the Streamlit interface

    logo_path = "ib_logo 1.jpg"
    with open(logo_path, "rb") as logo_file:
        logo_data = base64.b64encode(logo_file.read()).decode()
    logo_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_data}" style="max-width: 600px; max-height: 600px; display: block; margin: auto;" /></div>'
    st.markdown(logo_html, unsafe_allow_html=True)

    header_html = '<p style="color:#1a7fe3; font-size: 24px; text-align: center;">Interview Reviewer</p>'
    st.markdown(header_html, unsafe_allow_html=True)

    languages = {
        "English": "en",
        "Russian": "ru",
        # Add other languages
    }


    selected_original_language = st.selectbox("Select original language", list(languages.keys()))
    selected_translation_language = st.selectbox("Select translation language", list(languages.keys()))

    original_language = languages[selected_original_language]
    translation_language = languages[selected_translation_language]

    original_audio_file = st.file_uploader("Upload original audio file", type=["mp3"])
    translated_audio_file = st.file_uploader("Upload translated audio file", type=["mp3"])

    if original_audio_file is not None and translated_audio_file is not None:
        st.markdown("### Transcription in progress...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_original_audio_path = os.path.join(temp_dir, "original.mp3")
            with open(temp_original_audio_path, "wb") as f:
                f.write(original_audio_file.getbuffer())

            temp_translated_audio_path = os.path.join(temp_dir, "translated.mp3")
            with open(temp_translated_audio_path, "wb") as f:
                f.write(translated_audio_file.getbuffer())

            original_output_file = transcribe_audio(wisp_model, temp_original_audio_path, "original_transcription.txt",
                                                    original_language)
            translated_output_file = transcribe_audio(wisp_model, temp_translated_audio_path, "translated_transcription.txt",
                                                      translation_language)

        st.markdown("### Transcription completed!")

        with open(original_output_file, "r") as f:
            original_transcription_result = f.read()
        st.text_area("Original Transcription", original_transcription_result)

        with open(translated_output_file, "r") as f:
            translation_transcription_result = f.read()
        st.text_area("Translated Transcription", translation_transcription_result)

        # Evaluate the translation using GPT-4
        evaluation = evaluate_translation(original_transcription_result, translation_transcription_result,
                                          original_language, translation_language)

        st.markdown("### Translation evaluation by OPENAI GPT:")
        st.text_area("Evaluation", evaluation)

        # Allow user to download the evaluation result
        st.markdown("### Download evaluation as a text file")
        st.download_button(
            label="Download Evaluation",
            data=evaluation.encode("utf-8"),
            file_name="evaluation.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()
