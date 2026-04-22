"""
llm_client.py
-------------
HTTP client for Ollama's /api/generate endpoint.

Two modes:
  - live:  fast, small model, rolling summary of the last 2 minutes
  - post:  larger model, full analysis of the complete transcript

Both support streaming responses so the UI can show text as it arrives.
"""

import json
import time
import threading
import httpx
from typing import Iterator, Callable, Optional

from config import OLLAMA_BASE_URL, LIVE_MODEL, POST_MODEL, LIVE_SUMMARY_EVERY


# ── Prompt templates ───────────────────────────────────────────────────────

LIVE_SUMMARY_PROMPT = """\
You are a real-time meeting assistant. The following is a partial transcript \
from an ongoing meeting. Summarise what has been discussed in 3-5 bullet points. \
Be concise. Focus on key points, decisions, and unresolved questions.
Note any topic changes.

TRANSCRIPT:
{transcript}

SUMMARY:"""

POST_SUMMARY_PROMPT = """\
You are an expert meeting analyst. Below is the full transcript of a meeting.

Provide:
1. **Overall summary** (2-3 paragraphs)
2. **Key decisions made** (bullet list)
3. **Action items** — list each as: [Owner] Task description
4. **Topics discussed** — in chronological order with approximate timestamps
5. **Points of disagreement or open questions**

TRANSCRIPT:
{transcript}

ANALYSIS:"""

QA_PROMPT = """\
You are a meeting assistant. Answer the user's question using ONLY the \
information from the meeting transcript provided. If the answer is not \
in the transcript, say so clearly.

TRANSCRIPT EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""


class LLMClient:
    """Wraps Ollama's HTTP API for text generation."""

    def __init__(self):
        self._client = httpx.Client(timeout=120.0)

    def is_available(self) -> bool:
        """Check if Ollama is running and reachable."""
        try:
            r = self._client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    def stream(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream tokens from Ollama one by one.
        Yields string tokens as they arrive.
        """
        payload = {
            "model":  model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,    # factual summaries need low temp
                "num_predict": 1024,
            },
        }
        if system:
            payload["system"] = system

        with self._client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if token := data.get("response", ""):
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    def generate(self, prompt: str, model: str) -> str:
        """Non-streaming generation — returns the full response string."""
        return "".join(self.stream(prompt, model))

    # ── Live summary ───────────────────────────────────────────────────────

    def live_summary(self, transcript_text: str) -> str:
        """Generate a rolling summary of recent transcript text."""
        if not transcript_text.strip():
            return ""
        prompt = LIVE_SUMMARY_PROMPT.format(transcript=transcript_text)
        return self.generate(prompt, LIVE_MODEL)

    # ── Post-session ───────────────────────────────────────────────────────

    def post_analysis(self, full_transcript: str) -> Iterator[str]:
        """Stream a full post-session analysis. Returns a token iterator."""
        prompt = POST_SUMMARY_PROMPT.format(transcript=full_transcript)
        yield from self.stream(prompt, POST_MODEL)

    def answer_question(self, question: str, context: str) -> Iterator[str]:
        """Stream an answer to a user question given retrieved context."""
        prompt = QA_PROMPT.format(context=context, question=question)
        yield from self.stream(prompt, POST_MODEL)


# ── Live summary background worker ────────────────────────────────────────

class LiveSummaryWorker:
    """
    Background thread that refreshes the live summary every N seconds.
    Calls session_manager.update_live_summary() when a new summary is ready.
    """

    def __init__(self, llm: LLMClient, session_manager):
        self._llm     = llm
        self._session = session_manager
        self._thread  : Optional[threading.Thread] = None
        self._stop    = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[LiveSummary] started — refreshing every {LIVE_SUMMARY_EVERY}s")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        print("[LiveSummary] stopped")

    def _run(self) -> None:
        while not self._stop.wait(timeout=LIVE_SUMMARY_EVERY):
            try:
                recent = self._session.get_recent_text(last_n_seconds=120)
                if recent.strip():
                    summary = self._llm.live_summary(recent)
                    if summary:
                        self._session.update_live_summary(summary)
            except Exception as e:
                print(f"[LiveSummary] error: {e}")
