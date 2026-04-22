# MeetingMind — AI-Powered Meeting Intelligence System

Local-first meeting transcription, speaker diarization, and LLM-powered analysis. Runs entirely on your laptop — no cloud, no subscriptions.

## Overview

MeetingMind is a multi-stage pipeline that captures audio, converts speech to text, assigns speaker identities, generates live summaries, and enables post-session querying via a retrieval-augmented generation (RAG) system.

## Stack

| Layer         | Technology                         |
| ------------- | ---------------------------------- |
| Audio capture | `sounddevice`                      |
| Transcription | `faster-whisper` (CTranslate2)     |
| Diarization   | `pyannote/speaker-diarization-3.1` |
| LLM           | Ollama (`qwen2.5:3b` by default)   |
| Backend       | FastAPI + WebSocket                |
| Vector store  | ChromaDB                           |
| Frontend      | Plain HTML/JS (no build step)      |

## Architecture

Audio Input
→ Audio Capture (chunking with overlap)
→ Transcription (Faster-Whisper)
→ Diarization (pyannote)
→ Speaker Resolution
→ Session Management
→ LLM Processing (live + post)
→ RAG Indexing
→ WebSocket Delivery to UI

Each stage is modular and can be tested independently.

<img width="600" height="600" alt="meetingmind_architecture" src="https://github.com/user-attachments/assets/b2e29c46-5c16-4d8a-a0d0-7580a0188f4f" />

## Setup

### 1. Clone and create environment

```bash
cd meetingmind
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**HF Token setup (one-time):**

1. Create a free account at [https://huggingface.co](https://huggingface.co)
2. Accept model terms at [https://hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
3. Accept model terms at [https://hf.co/pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0)
4. Generate a token at [https://hf.co/settings/tokens](https://hf.co/settings/tokens)
5. Paste it into `.env` as `HF_TOKEN=hf_...`

### 3. Pull Ollama models

```bash
ollama pull qwen2.5:3b          # live summaries
ollama pull nomic-embed-text    # RAG embeddings (post-session Q&A)

# Optional: larger model for post-session analysis
ollama pull llama3.1:8b
```

## Running

```bash
cd backend
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Enter known speaker names (comma-separated)
2. Press Start to begin recording
3. Speak; transcript appears in real time with speaker labels
4. Live summary refreshes periodically
5. Press Stop when the meeting ends
6. Press Analyse for post-session insights
7. Use the Q&A tab to query the transcript

### Renaming speakers

Rename any anonymous speaker ("Speaker N") from the UI.

## Testing individual components

```bash
cd backend
python audio_capture.py
python transcriber.py path/to/audio.wav
python diarizer.py path/to/audio.wav
```

## Project Structure

```text
meetingmind/
├── backend/
│   ├── main.py              # FastAPI app + WebSocket
│   ├── audio_capture.py     # Mic input → chunking
│   ├── transcriber.py       # faster-whisper wrapper
│   ├── diarizer.py          # pyannote diarization
│   ├── speaker_registry.py  # Cross-chunk speaker identity
│   ├── session_manager.py   # Session state + persistence
│   ├── llm_client.py        # Ollama client (live + post)
│   ├── rag.py               # RAG pipeline
│   └── config.py            # Configuration
├── frontend/
│   └── index.html           # Single-file UI
├── sessions/                # Stored transcripts
├── requirements.txt
└── .env
```

## Performance Tips

| Machine         | Recommended Whisper model | Notes            |
| --------------- | ------------------------- | ---------------- |
| MacBook (Intel) | `base` or `tiny`          | CPU-only         |
| GPU system      | `small` or `medium`       | Faster inference |

Adjust in `backend/config.py`.

## Limitations

* Near-live processing introduces latency
* Diarization consistency can drift across long sessions
* CPU inference may be slow on low-end machines
* Single active session at a time

## Summary

MeetingMind converts conversations into structured, queryable data using speech recognition, speaker diarization, language models, and vector search. It is designed as a complete system blueprint with clear extension points for future development.
