"""
session_manager.py
------------------
Central state object for a single meeting session.

Owns:
  - the growing transcript (list of segments)
  - session metadata (start time, topic, attendees)
  - file persistence (JSON + plain text)
  - broadcast hook for WebSocket push (set by main.py)
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from config import SESSIONS_DIR


class SessionManager:

    def __init__(self, session_id: Optional[str] = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id   = session_id or ts
        self.session_dir  = SESSIONS_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.started_at   = time.time()
        self.transcript   : list[dict] = []   # all segments so far
        self.live_summary : str = ""
        self.topics       : list[dict] = []   # [{time, topic}]
        self.speakers     : list[str]  = []   # known display names

        # Called by main.py to push updates to connected WebSocket clients
        self._broadcast_fn: Optional[Callable] = None

        print(f"[Session] started — id={self.session_id}")

    # ── Broadcast hook ────────────────────────────────────────────────────

    def set_broadcast(self, fn: Callable) -> None:
        self._broadcast_fn = fn

    def _broadcast(self, event: dict) -> None:
        if self._broadcast_fn:
            self._broadcast_fn(event)

    # ── Transcript updates ────────────────────────────────────────────────

    def add_segments(self, segments: list[dict]) -> None:
        """
        Append new transcript segments and broadcast them to the UI.

        Each segment must have at minimum:
            start, end, text, speaker
        """
        if not segments:
            return

        for seg in segments:
            seg.setdefault("speaker", "Unknown")
            self.transcript.append(seg)

            # Track new speaker names
            spk = seg["speaker"]
            if spk not in self.speakers:
                self.speakers.append(spk)

        # Broadcast each new segment to connected clients
        self._broadcast({
            "event":    "transcript",
            "segments": segments,
        })

        # Auto-save after each batch
        self._save()

    def update_live_summary(self, summary: str) -> None:
        """Update the rolling live summary and push to UI."""
        self.live_summary = summary
        self._broadcast({
            "event":   "live_summary",
            "summary": summary,
        })
        self._save()

    def add_topic(self, topic: str, at_time: Optional[float] = None) -> None:
        """Log a detected topic change."""
        entry = {
            "time":  at_time or (time.time() - self.started_at),
            "topic": topic,
        }
        self.topics.append(entry)
        self._broadcast({
            "event": "topic",
            **entry,
        })

    # ── Queries ──────────────────────────────────────────────────────────

    def get_recent_text(self, last_n_seconds: float = 120) -> str:
        """
        Return the transcript text from the last N seconds.
        Used by the live LLM summariser to avoid re-processing the full transcript.
        """
        cutoff = (time.time() - self.started_at) - last_n_seconds
        recent = [s for s in self.transcript if s.get("start", 0) >= cutoff]
        return self._format_transcript(recent)

    def get_full_text(self) -> str:
        """Return the entire transcript formatted as plain text."""
        return self._format_transcript(self.transcript)

    def get_state_snapshot(self) -> dict:
        """
        Full state snapshot — sent to a newly connected WebSocket client
        so they can catch up on everything missed before connecting.
        """
        return {
            "event":        "snapshot",
            "session_id":   self.session_id,
            "started_at":   self.started_at,
            "transcript":   self.transcript,
            "live_summary": self.live_summary,
            "topics":       self.topics,
            "speakers":     self.speakers,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        """Write current session state to disk (JSON + plain text)."""
        # JSON — machine-readable, used by post-session pipeline and RAG
        json_path = self.session_dir / "transcript.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id":   self.session_id,
                "started_at":   self.started_at,
                "speakers":     self.speakers,
                "topics":       self.topics,
                "live_summary": self.live_summary,
                "transcript":   self.transcript,
            }, f, ensure_ascii=False, indent=2)

        # Plain text — human readable, easy to share
        txt_path = self.session_dir / "transcript.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Speakers: {', '.join(self.speakers)}\n")
            f.write("─" * 60 + "\n\n")
            f.write(self._format_transcript(self.transcript))
            if self.live_summary:
                f.write("\n\n" + "─" * 60 + "\nSummary\n" + "─" * 60 + "\n")
                f.write(self.live_summary + "\n")

    def finalise(self) -> dict:
        """
        Mark session as complete and return paths to saved files.
        Call this when the meeting ends.
        """
        self._save()
        return {
            "session_id":    self.session_id,
            "session_dir":   str(self.session_dir),
            "transcript_json": str(self.session_dir / "transcript.json"),
            "transcript_txt":  str(self.session_dir / "transcript.txt"),
            "duration_seconds": time.time() - self.started_at,
            "speakers":      self.speakers,
            "segment_count": len(self.transcript),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(segments: list[dict]) -> str:
        lines = []
        prev_speaker = None
        for seg in segments:
            spk  = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            t    = seg.get("start", 0)
            mins, secs = divmod(int(t), 60)
            timestamp  = f"[{mins:02d}:{secs:02d}]"

            if spk != prev_speaker:
                lines.append(f"\n{timestamp} {spk}:")
                prev_speaker = spk

            lines.append(f"  {text}")

        return "\n".join(lines).strip()
