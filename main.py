import streamlit as st
import openai
import pydub
import os
from io import BytesIO
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function to convert speech to text
def speech_to_text(audio_file):
    audio_file.seek(0)  # Make sure the file is at the beginning
    transcript = openai.Audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )
    return transcript['text']

# Function to generate OpenAI response based on text input
def get_openai_response(text):
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can choose other models like gpt-3.5-turbo
        prompt=text,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# Function to convert text to speech and return the audio
def text_to_speech(text):
    response = openai.Audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    # Convert the response stream to an MP3 file
    audio_data = BytesIO(response.content)
    return audio_data

# Streamlit App
st.title("Speech to Text Chat App")

# Record audio section
st.header("Step 1: Record Your Voice")

audio_data = st.file_uploader("Upload an audio file (MP3/WAV format)", type=["mp3", "wav"])

if audio_data:
    # Convert speech to text
    st.header("Step 2: Transcribing Speech to Text...")
    with st.spinner('Transcribing...'):
        text_input = speech_to_text(audio_data)
    st.write("**Transcribed Text:**", text_input)

    # Get OpenAI response based on text input
    st.header("Step 3: Get OpenAI Response")
    with st.spinner('Getting response from OpenAI...'):
        openai_response = get_openai_response(text_input)
    st.write("**OpenAI Response:**", openai_response)

    # Convert OpenAI response to speech
    st.header("Step 4: Converting Response to Speech...")
    with st.spinner('Converting text to speech...'):
        speech_response = text_to_speech(openai_response)

    # Play the audio response
    st.audio(speech_response, format="audio/mp3")
