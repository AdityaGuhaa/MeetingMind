"""
main.py
-------
FastAPI application — the central orchestrator.

Endpoints:
  POST /session/start          start a new session (registers known speakers)
  POST /session/stop           stop recording, finalise session
  GET  /session/{id}/status    current session state
  POST /session/{id}/rename    rename a speaker
  POST /session/{id}/analyse   trigger post-session full analysis
  POST /session/{id}/ask       Q&A against the session transcript
  GET  /sessions               list all past sessions
  WS   /ws                     real-time updates (transcript, summary, topics)
  GET  /                       serves the frontend HTML
"""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import WS_HOST, WS_PORT, SESSIONS_DIR
from audio_capture import AudioCapture
from transcriber import Transcriber
from diarizer import Diarizer
from speaker_registry import SpeakerRegistry
from session_manager import SessionManager
from llm_client import LLMClient, LiveSummaryWorker
from rag import RAGIndex


# ── Global state ──────────────────────────────────────────────────────────

active_session  : Optional[SessionManager]   = None
audio_capture   : Optional[AudioCapture]     = None
transcriber     : Transcriber                = Transcriber()
diarizer        : Diarizer                   = Diarizer()
registry        : Optional[SpeakerRegistry]  = None
llm             : LLMClient                  = LLMClient()
live_worker     : Optional[LiveSummaryWorker] = None
rag_index       : Optional[RAGIndex]         = None
processing_thread: Optional[threading.Thread] = None
stop_processing  = threading.Event()

# WebSocket connection pool
ws_clients: list[WebSocket] = []


# ── Startup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[MeetingMind] loading models...")
    transcriber.load()
    diarizer.load()
    print("[MeetingMind] ready")
    yield
    print("[MeetingMind] shutting down")


app = FastAPI(title="MeetingMind", lifespan=lifespan)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if (FRONTEND_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


# ── WebSocket ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    print(f"[WS] client connected — {len(ws_clients)} total")

    # Send full state snapshot so the client catches up immediately
    if active_session:
        await ws.send_text(json.dumps(active_session.get_state_snapshot()))

    try:
        while True:
            await ws.receive_text()   # keep alive; UI sends no messages
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        print(f"[WS] client disconnected — {len(ws_clients)} remaining")


def broadcast(event: dict) -> None:
    """
    Called from non-async threads to push events to all WS clients.
    Uses asyncio.run_coroutine_threadsafe since WS sending is async.
    """
    if not ws_clients:
        return
    loop = asyncio.get_event_loop()
    msg  = json.dumps(event)
    for ws in list(ws_clients):
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)
        except Exception:
            pass


# ── Request models ────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    known_speakers: list[str] = []   # e.g. ["Aditya", "Rohan", "Priya"]

class RenameRequest(BaseModel):
    current_label: str
    new_label:     str

class AskRequest(BaseModel):
    question: str


# ── Session lifecycle ─────────────────────────────────────────────────────

@app.post("/session/start")
async def start_session(req: StartRequest):
    global active_session, audio_capture, registry, live_worker
    global processing_thread, stop_processing, rag_index

    if active_session:
        raise HTTPException(400, "A session is already running. Stop it first.")

    # Initialise session objects
    active_session = SessionManager()
    active_session.set_broadcast(broadcast)

    registry = SpeakerRegistry()
    registry.load_embedding_model()
    for name in req.known_speakers:
        registry.register(name)

    rag_index = RAGIndex(active_session.session_id)

    # Start audio capture
    audio_capture = AudioCapture()
    audio_capture.start()

    # Start background processing thread
    stop_processing.clear()
    processing_thread = threading.Thread(
        target=_processing_loop,
        args=(audio_capture, active_session, registry),
        daemon=True,
    )
    processing_thread.start()

    # Start live summary worker
    live_worker = LiveSummaryWorker(llm, active_session)
    live_worker.start()

    return {
        "status":     "started",
        "session_id": active_session.session_id,
        "speakers":   req.known_speakers,
    }


@app.post("/session/stop")
async def stop_session():
    global active_session, audio_capture, live_worker, processing_thread

    if not active_session:
        raise HTTPException(400, "No active session.")

    # Stop all workers in order
    stop_processing.set()
    if audio_capture:
        audio_capture.stop()
    if live_worker:
        live_worker.stop()
    if processing_thread:
        processing_thread.join(timeout=10)

    result = active_session.finalise()

    # Reset globals
    active_session    = None
    audio_capture     = None
    live_worker       = None
    processing_thread = None

    return result


@app.get("/session/{session_id}/status")
async def session_status(session_id: str):
    if active_session and active_session.session_id == session_id:
        return active_session.get_state_snapshot()
    # Try loading from disk
    json_path = SESSIONS_DIR / session_id / "transcript.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    raise HTTPException(404, "Session not found")


@app.post("/session/{session_id}/rename")
async def rename_speaker(session_id: str, req: RenameRequest):
    if not registry:
        raise HTTPException(400, "No active session registry.")
    ok = registry.rename(req.current_label, req.new_label)
    if not ok:
        raise HTTPException(404, f"Speaker '{req.current_label}' not found.")
    return {"status": "ok", "new_label": req.new_label}


@app.post("/session/{session_id}/analyse")
async def post_analysis(session_id: str):
    """Run full post-session analysis. Streams back the result."""
    json_path = SESSIONS_DIR / session_id / "transcript.json"
    if not json_path.exists():
        raise HTTPException(404, "Session transcript not found.")

    with open(json_path) as f:
        data = json.load(f)

    session = SessionManager(session_id)
    session.transcript = data.get("transcript", [])
    full_text = session.get_full_text()

    if not full_text.strip():
        raise HTTPException(400, "Transcript is empty.")

    # Index into RAG
    if rag_index and rag_index.session_id == session_id:
        rag_index.index_transcript(session.transcript)

    analysis = "".join(llm.post_analysis(full_text))

    # Persist analysis to disk
    analysis_path = SESSIONS_DIR / session_id / "analysis.md"
    with open(analysis_path, "w") as f:
        f.write(f"# MeetingMind Analysis — {session_id}\n\n")
        f.write(analysis)

    return {"analysis": analysis, "saved_to": str(analysis_path)}


@app.post("/session/{session_id}/ask")
async def ask_question(session_id: str, req: AskRequest):
    """Answer a question using RAG over the session transcript."""
    if not rag_index or rag_index.session_id != session_id:
        raise HTTPException(400, "RAG index not ready. Run /analyse first.")

    context = rag_index.retrieve(req.question)
    answer  = "".join(llm.answer_question(req.question, context))
    return {"question": req.question, "answer": answer}


@app.get("/sessions")
async def list_sessions():
    """List all saved sessions."""
    sessions = []
    for d in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        json_path = d / "transcript.json"
        if json_path.exists():
            with open(json_path) as f:
                meta = json.load(f)
            sessions.append({
                "session_id":    meta.get("session_id"),
                "started_at":    meta.get("started_at"),
                "speakers":      meta.get("speakers", []),
                "segment_count": len(meta.get("transcript", [])),
            })
    return sessions


# ── Frontend ──────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>MeetingMind</h1><p>Frontend not found.</p>")


# ── Processing loop (runs in background thread) ───────────────────────────

def _processing_loop(
    capture: AudioCapture,
    session: SessionManager,
    reg: SpeakerRegistry,
) -> None:
    """
    Pulls audio chunks from the capture queue, runs transcription
    and diarization, resolves speaker names, and adds to session.
    """
    print("[Processing] loop started")
    while not stop_processing.is_set():
        chunk = capture.get_chunk(timeout=1.0)
        if chunk is None:
            continue

        audio      = chunk["audio"]
        start_time = chunk["start_time"]

        try:
            # Step 1: transcribe
            segments = transcriber.transcribe(audio, chunk_start_time=start_time)
            if not segments:
                continue

            # Step 2: diarize
            diar_segs = diarizer.diarize(audio)

            # Step 3: assign raw speaker IDs to segments
            segments = diarizer.assign_speakers(segments, diar_segs, start_time)

            # Step 4: resolve to display names
            segments = reg.resolve_segments(segments, audio, diar_segs)

            # Step 5: add to session (triggers broadcast + file save)
            session.add_segments(segments)

        except Exception as e:
            print(f"[Processing] error on chunk at t={start_time:.1f}s: {e}")

    print("[Processing] loop stopped")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=WS_HOST,
        port=WS_PORT,
        reload=False,
        log_level="info",
    )
