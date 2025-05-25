import sqlite3
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
import re
from datetime import datetime

def split_text(text, max_length=1000):
    # Split text into sentences, then group into chunks <= max_length
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ''
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_length:
            current += (' ' if current else '') + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def synthesize_summary_to_mp3(db_path: str, output_path: str):
    # Connect to the SQLite database and get the last summary
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM llm_responses ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if not row or not row[0]:
        raise ValueError("No summary found in the database.")
    summary_text = row[0]

    # Split summary into chunks for gTTS
    chunks = split_text(summary_text, max_length=1000)
    audio_segments = []
    temp_files = []
    for i, chunk in enumerate(chunks):
        tts = gTTS(text=chunk, lang='en', slow=False)
        with NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
            tts.save(tf.name)
            temp_files.append(tf.name)
            audio_segments.append(AudioSegment.from_mp3(tf.name))
    # Concatenate all audio segments
    combined = audio_segments[0]
    for seg in audio_segments[1:]:
        combined += seg
    combined.export(output_path, format="mp3")
    # Clean up temp files
    for tf in temp_files:
        os.remove(tf)
    return output_path

def generate_speech_from_text(text: str, output_dir: str = "backend/"):
    """
    Generate speech from text using gTTS, save as MP3, and return the file path in a dict.
    """
    if not text or not text.strip():
        return {"audio_content_url": None, "message": "No text provided for TTS."}
    # Clean up text for TTS
    text = text.replace('—', '-')
    text = text.replace('..', '.')
    text = text.replace('  ', ' ')
    text = text.strip()
    # Generate a unique filename
    filename = f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    output_path = os.path.join(output_dir, filename)
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    return {"audio_content_url": output_path, "message": "TTS generated successfully."}

# Example usage (uncomment to run directly):
# synthesize_summary_to_mp3(
#     db_path="/home/timc/Documents/github/FinanceGuru/backend/llm_interactions.sqlite",
#     output_path="/home/timc/Documents/github/FinanceGuru/backend/tests/llm_summary.mp3"
# )
