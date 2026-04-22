# MeetingMind

Local-first meeting transcription, speaker diarization, and LLM-powered analysis.
Runs entirely on your laptop — no cloud, no subscriptions.

---

## Stack

| Layer | Technology |
|---|---|
| Audio capture | `sounddevice` |
| Transcription | `faster-whisper` (CTranslate2) |
| Diarization | `pyannote/speaker-diarization-3.1` |
| LLM | Ollama (`qwen2.5:3b` by default) |
| Backend | FastAPI + WebSocket |
| Vector store | ChromaDB |
| Frontend | Plain HTML/JS (no build step) |

---

## Setup

### 1. Clone and create environment

```bash
cd meetingmind
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

**HF Token setup (one-time):**
1. Create a free account at https://huggingface.co
2. Accept model terms at https://hf.co/pyannote/speaker-diarization-3.1
3. Accept model terms at https://hf.co/pyannote/segmentation-3.0
4. Generate a token at https://hf.co/settings/tokens
5. Paste it into `.env` as `HF_TOKEN=hf_...`

### 3. Pull Ollama models

```bash
ollama pull qwen2.5:3b          # live summaries
ollama pull nomic-embed-text    # RAG embeddings (post-session Q&A)

# Optional: larger model for post-session analysis
ollama pull llama3.1:8b
```

---

## Running

```bash
cd backend
python main.py
```

Open **http://localhost:8000** in your browser.

---

## Usage

1. Enter known speaker names (comma-separated) in the input box
2. Press **Start** — recording begins immediately
3. Speak — transcript appears in real-time, attributed to speakers
4. Live summary refreshes every 30 seconds in the right panel
5. Press **Stop** when the meeting ends
6. Press **Analyse** for full post-session analysis
7. Use the **Q&A** tab to query the transcript: *"What did Rohan say about budget?"*

### Renaming anonymous speakers
Click **rename** next to any "Speaker N" label in the Speakers tab.

---

## Testing individual components

```bash
# Test audio capture (watches mic for 30s, prints chunk stats)
cd backend
python audio_capture.py

# Test transcription (pass a WAV file)
python transcriber.py path/to/audio.wav

# Test diarization (pass a WAV file)
python diarizer.py path/to/audio.wav
```

---

## Performance tips

| Machine | Recommended Whisper model | Notes |
|---|---|---|
| MacBook (Intel, no Metal) | `base` or `tiny` | CPU-only; base ≈ 2–3× realtime |
| Lenovo LOQ (NVIDIA GPU) | `small` or `medium` | CUDA speeds this up ~10× |

Switch the model in `backend/config.py`:
```python
WHISPER_MODEL = "small"   # tiny | base | small | medium | large-v3
```

---

## Project structure

```
meetingmind/
├── backend/
│   ├── main.py              FastAPI app + WebSocket
│   ├── audio_capture.py     Mic → 15s chunks
│   ├── transcriber.py       faster-whisper wrapper
│   ├── diarizer.py          pyannote diarization
│   ├── speaker_registry.py  Cross-chunk speaker identity
│   ├── session_manager.py   Session state + file persistence
│   ├── llm_client.py        Ollama client (live + post)
│   ├── rag.py               ChromaDB RAG for Q&A
│   └── config.py            All tunable parameters
├── frontend/
│   └── index.html           Single-file web UI
├── sessions/                Saved transcripts (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```
