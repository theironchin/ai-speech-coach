import streamlit as st
import openai
import whisper
import numpy as np
import tempfile

# Configure your OpenAI API key
openai.api_key = 'YOUR_API_KEY_HERE'

# Load Whisper model
model = whisper.load_model('base')

st.title("🎙️ AI Speech Coach")
st.write("Record your speech, and receive advanced coaching powered by Whisper and GPT-4.")

# Words to track
filler_words = ["if", "but", "I'm", "um", "uh", "like", "so", "you know"]

def analyze_pacing(text, duration_sec):
    words = text.split()
    wpm = (len(words) / duration_sec) * 60
    return round(wpm, 2)

uploaded_file = st.file_uploader("Record and upload your speech (MP3 or WAV):", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tmp.write(uploaded_file.read())
        audio_file = tmp.name
        
        # Transcribe using Whisper
        st.write("Transcribing your speech...")
        result = model.transcribe(audio_file)
        speech_text = result["text"]
        duration_sec = result['segments'][-1]['end']

        st.subheader("Your Speech Transcript")
        st.write(speech_text)

        # Count filler words
        word_counts = {word: speech_text.lower().split().count(word) for word in filler_words}
        
        # Analyze pacing
        wpm = analyze_pacing(speech_text, duration_sec)

        # Create prompt for GPT feedback
        prompt = f"""
        You're an expert speech coach. Analyze this speech:

        Transcript:
        {speech_text}

        Filler word counts: {word_counts}
        Pace: {wpm} words per minute.

        Provide:
        1. Feedback on filler words.
        2. Feedback on pacing.
        3. Suggestions to improve clarity and confidence.
        """

        # Get GPT feedback
        st.write("Generating AI coaching feedback...")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        feedback = response.choices[0].message.content

        # Display results
        st.subheader("Your Speech Transcript")
        st.write(speech_text)

        st.subheader("Analysis Results")
        st.write(f"**Words Per Minute (WPM):** {wpm}")

        st.subheader("Filler Words Usage:")
        for word, count in word_counts.items():
            st.write(f"- **{word}:** {count} times")

        st.subheader("AI Coaching Feedback:")
        st.write(response.choices[0].message.content)
