import json
import sqlite3
import tempfile
import wave
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel
from scipy.io.wavfile import write as wav_write


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "niko_memory.db"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

SYSTEM_PROMPT = """
You are Niko, a friendly personal AI assistant.
My name is Zainab greet me every time you run.
You speak naturally and concisely.
The user may speak Egyptian Arabic, English, or a mix of both.
Reply in the same language the user mainly uses.
Do not claim you performed an action unless the application actually provides that tool.
You have access to a small memory database containing facts the user explicitly asked you to remember.
"""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL UNIQUE
        )
    """)
    conn.commit()
    conn.close()


def get_memories():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT memory FROM memories ORDER BY id DESC LIMIT 30").fetchall()
    conn.close()
    return [row[0] for row in rows]


def save_memory(memory):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("INSERT OR IGNORE INTO memories(memory) VALUES (?)", (memory,))
        conn.commit()
    finally:
        conn.close()


def record_audio(seconds=RECORD_SECONDS):
    print(f"\nMIC: Speak now ({seconds} seconds)...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    path = Path(tempfile.gettempdir()) / "niko_input.wav"
    wav_write(path, SAMPLE_RATE, audio)
    return path


def transcribe(model, audio_path):
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def ask_ollama(user_text):
    memories = get_memories()

    memory_block = "\n".join(f"- {m}" for m in memories)
    if not memory_block:
        memory_block = "(No saved memories.)"

    prompt = f"""
Saved memories:
{memory_block}

User:
{user_text}

If the user explicitly asks you to remember something, reply normally and include:
MEMORY_TO_SAVE: <short fact>
at the end of your answer.
Otherwise do not include MEMORY_TO_SAVE.
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def extract_memory(reply):
    marker = "MEMORY_TO_SAVE:"
    if marker not in reply:
        return reply

    before, after = reply.split(marker, 1)
    memory = after.strip().splitlines()[0].strip()
    if memory:
        save_memory(memory)

    return before.strip()


# def speak(engine, text):
#     print(f"\nNiko: {text}")
#     engine.say(text)
#     engine.runAndWait()
def speak(engine, text):
    print(f"\nNiko: {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def main():
    init_db()

    print("Loading Whisper model...")
    # Small model is a reasonable starting point for CPU.
    whisper = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
    )

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    print("\n==============================")
    print("       NIKO v1 READY")
    print("==============================")
    print("Press Enter to talk.")
    print("Type q then Enter to quit.")

    while True:
        command = input("\n> ")
        if command.lower().strip() == "q":
            break

        try:
            audio_path = record_audio()
            user_text = transcribe(whisper, audio_path)

            if not user_text:
                print("I didn't hear anything.")
                continue

            print(f"You: {user_text}")

            reply = ask_ollama(user_text)
            reply = extract_memory(reply)

            speak(engine, reply)

        except requests.exceptions.ConnectionError:
            print(
                "\nOllama is not reachable. Start Ollama and make sure "
                f"the model '{OLLAMA_MODEL}' is installed."
            )
        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
