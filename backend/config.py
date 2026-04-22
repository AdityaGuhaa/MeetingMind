import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

# ── Audio capture ──────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000   # Hz — Whisper expects 16 kHz mono
CHUNK_SECONDS     = 15      # near-live window size
OVERLAP_SECONDS   = 2       # overlap between chunks for continuity
CHANNELS          = 1       # mono

# ── Whisper ────────────────────────────────────────────────────────────────
WHISPER_MODEL     = "base"  # tiny | base | small | medium | large-v3
                            # base is the sweet spot for near-live on CPU
                            # switch to small/medium for post-session pass
WHISPER_DEVICE    = "auto"  # "cpu" | "cuda" | "auto"
WHISPER_COMPUTE   = "int8"  # int8 keeps RAM low on laptop CPU

# ── Diarization ────────────────────────────────────────────────────────────
HF_TOKEN          = os.getenv("HF_TOKEN", "")
DIARIZE_MODEL     = "pyannote/speaker-diarization-3.1"
MAX_SPEAKERS      = 10      # upper bound — pyannote will detect the actual count

# ── Ollama LLM ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LIVE_MODEL        = os.getenv("LIVE_MODEL",  "qwen2.5:3b")   # fast, rolling summary
POST_MODEL        = os.getenv("POST_MODEL",  "qwen2.5:3b")   # swap to 14b if RAM allows
LIVE_SUMMARY_EVERY = 30     # seconds between live summary refreshes

# ── RAG / ChromaDB ─────────────────────────────────────────────────────────
CHROMA_DIR        = BASE_DIR / "sessions" / ".chromadb"
EMBED_MODEL       = "nomic-embed-text"   # pulled via: ollama pull nomic-embed-text
CHUNK_SIZE        = 400     # characters per RAG chunk
CHUNK_OVERLAP     = 80

# ── WebSocket ──────────────────────────────────────────────────────────────
WS_HOST           = "0.0.0.0"
WS_PORT           = 8000
