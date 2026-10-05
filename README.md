# Niko v1

A local-first voice AI assistant prototype.

Pipeline:
Microphone -> Faster-Whisper (local STT) -> Ollama (local LLM) -> SQLite memory -> pyttsx3 (local TTS)

## Requirements
- Windows 10/11
- Python 3.11 recommended
- Ollama installed and running
- A microphone + speakers

## 1) Create environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Install an Ollama model
```bash
ollama pull llama3.2:3b
```

If your PC has enough RAM/VRAM, you can use a larger model.

## 3) Run
```bash
python niko.py
```

Say something, wait for transcription, and Niko will answer.

## Notes
- Speech recognition is local through Faster-Whisper.
- LLM inference is local through Ollama.
- Memory is stored in `niko_memory.db`.
- TTS is local through Windows SAPI via pyttsx3.
- This v1 does NOT control WhatsApp, Gmail, calls, contacts, or other phone data.
