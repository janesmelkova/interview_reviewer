import base64
import streamlit as st
import whisper
import tempfile

# Load the Whisper model
model = whisper.load_model("large") # change the model -- tbu add tickbox to the interface


def transcribe_audio(model, audio_path, output_file, language):
    # Load the audio file and transcribe it with progress bar and timer
    result = model.transcribe(
        audio_path,
        language=language,  # Change the language
        fp16=False,
        verbose=True,
    )

    with open(output_file, "w") as f:
        f.write(result["text"])

    print(f"Transcription results saved to {output_file}")
    return output_file

def main():

    logo_path = "ib_logo 1.jpg"  # Replace with your logo file name
    with open(logo_path, "rb") as logo_file:
        logo_data = base64.b64encode(logo_file.read()).decode()
    logo_html = f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_data}" style="max-width: 600px; max-height: 600px; display: block; margin: auto;" /></div>'
    st.markdown(logo_html, unsafe_allow_html=True)


    header_html = '<p style="color:#1a7fe3; font-size: 24px; text-align: center;">Audio Transcription</p>'
    st.markdown(header_html, unsafe_allow_html=True)


    languages = {
        "Russian": "ru",
        "English": "en",
        # Add more languages
    }

    selected_language = st.selectbox("Select language", list(languages.keys()))
    language_code = languages[selected_language]

    original_audio_file = st.file_uploader("Upload original audio file", type=["mp3", "wav"])
    translated_audio_file = st.file_uploader("Upload translated audio file", type=["mp3", "wav"])

    if original_audio_file is not None and translated_audio_file is not None:
        st.markdown("### Transcription in progress...")

        temp_original_audio_path = tempfile.mkstemp(suffix=".mp3")[1]
        with open(temp_original_audio_path, "wb") as f:
            f.write(original_audio_file.getbuffer())

        temp_translated_audio_path = tempfile.mkstemp(suffix=".mp3")[1]
        with open(temp_translated_audio_path, "wb") as f:
            f.write(translated_audio_file.getbuffer())

        original_output_file = transcribe_audio(model, temp_original_audio_path, "original_transcription.txt",
                                                language_code)
        translated_output_file = transcribe_audio(model, temp_translated_audio_path, "translated_transcription.txt",
                                                  language_code)

        st.markdown("### Transcription completed!")

        with open(original_output_file, "r") as f:
            original_transcription_result = f.read()
        st.text_area("Original Transcription", original_transcription_result)

        with open(translated_output_file, "r") as f:
            translated_transcription_result = f.read()
        st.text_area("Translated Transcription", translated_transcription_result)

        st.markdown("### Download transcriptions as text files")
        st.download_button(
            label="Download Original",
            data=open(original_output_file, "rb").read(),
            file_name="original_transcription.txt",
            mime="text/plain",
        )
        st.download_button(
            label="Download Translated",
            data=open(translated_output_file, "rb").read(),
            file_name="translated_transcription.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()