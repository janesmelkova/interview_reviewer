import base64
import streamlit as st
import whisper

# Load the Whisper model
model = whisper.load_model("large")


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


    #st.title('Transcribe your audio')
    header_html = '<p style="color:#1a7fe3; font-size: 24px; text-align: center;">Audio Transcription</p>'
    st.markdown(header_html, unsafe_allow_html=True)


    languages = {
        "Russian": "ru",
        "English": "en",
        # Add more languages if needed
    }

    selected_language = st.selectbox("Select language", list(languages.keys()))
    language_code = languages[selected_language]

    audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

    if audio_file is not None:
        st.markdown("### Transcription in progress...")

        # Save the uploaded audio file temporarily
        import os
        import tempfile

        temp_audio_path = tempfile.mkstemp(suffix=".mp3")[1]
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Transcription process
        output_file = transcribe_audio(model, temp_audio_path, "transcription.txt", language_code)

        st.markdown("### Transcription completed!")

        # Display the transcription result
        with open(output_file, "r") as f:
            transcription_result = f.read()
        st.text_area("Transcription", transcription_result)

        # Allow user to download the transcription result
        st.markdown("### Download transcription as a text file")
        st.download_button(
            label="Download",
            data=open(output_file, "rb").read(),
            file_name="transcription.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()











# Define the audio file path and output file path
#audio_path = "/Users/zhannasmelkova/Desktop/ИИ/Тест для собеседований переводчиков/2021-11-29-Gazprom-MO-RUS-3 min.mp3"
#output_file = "/Users/zhannasmelkova/Desktop/ИИ/Тест для собеседований переводчиков/2021-11-29-Gazprom-MO-RUS-3 min-1.txt"

# Transcription process
#transcribe_audio(model, audio_path, output_file)
