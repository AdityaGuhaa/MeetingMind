"""
audio_capture.py
----------------
Captures microphone audio in rolling chunks and yields raw numpy arrays.

Usage (standalone test):
    python audio_capture.py

Each chunk is CHUNK_SECONDS of audio at 16 kHz mono float32.
An OVERLAP_SECONDS tail from the previous chunk is prepended so Whisper
doesn't miss words at chunk boundaries.
"""

import threading
import queue
import time
import numpy as np

import sounddevice as sd

from config import SAMPLE_RATE, CHUNK_SECONDS, OVERLAP_SECONDS, CHANNELS


class AudioCapture:
    """
    Streams microphone audio and emits overlapping fixed-length chunks
    into an output queue.

    Each item placed on `self.chunk_queue` is a dict:
        {
            "audio":      np.ndarray  float32, shape (N,),
            "start_time": float       wall-clock seconds since session start,
            "end_time":   float
        }
    """

    def __init__(self):
        self.chunk_queue: queue.Queue = queue.Queue()
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None

        self._chunk_samples   = int(SAMPLE_RATE * CHUNK_SECONDS)
        self._overlap_samples = int(SAMPLE_RATE * OVERLAP_SECONDS)

        # Rolling buffer accumulates raw samples from the sounddevice callback
        self._buffer      = np.zeros(0, dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._session_start: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Begin capturing audio from the default microphone."""
        self._stop_event.clear()
        self._session_start = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[AudioCapture] started — {CHUNK_SECONDS}s chunks, "
              f"{OVERLAP_SECONDS}s overlap, {SAMPLE_RATE} Hz")

    def stop(self) -> None:
        """Stop capturing and flush any remaining audio as a final chunk."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

        # Flush remaining buffer as final chunk if it has meaningful content
        with self._buffer_lock:
            if len(self._buffer) > SAMPLE_RATE * 1:   # at least 1 second
                self._emit_chunk(self._buffer.copy(), is_final=True)
                self._buffer = np.zeros(0, dtype=np.float32)

        print("[AudioCapture] stopped")

    def get_chunk(self, timeout: float = 1.0) -> dict | None:
        """
        Blocking get from the chunk queue.
        Returns None on timeout — check stop_event and retry.
        """
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── Internal ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main thread: opens sounddevice stream and drains into _buffer."""
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.5),   # 500ms callback blocks
            callback=self._sd_callback,
        ):
            while not self._stop_event.is_set():
                time.sleep(0.1)
                self._try_emit()

    def _sd_callback(self, indata: np.ndarray, frames: int,
                     time_info, status) -> None:
        """sounddevice callback — runs on audio thread, must be fast."""
        if status:
            print(f"[AudioCapture] sounddevice status: {status}")
        samples = indata[:, 0].copy()   # mono: take channel 0
        with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, samples])

    def _try_emit(self) -> None:
        """Check if buffer has enough samples for a full chunk; emit if so."""
        with self._buffer_lock:
            if len(self._buffer) >= self._chunk_samples:
                chunk_audio = self._buffer[: self._chunk_samples].copy()
                # Keep overlap tail so next chunk has context
                self._buffer = self._buffer[
                    self._chunk_samples - self._overlap_samples :
                ].copy()
                self._emit_chunk(chunk_audio)

    def _emit_chunk(self, audio: np.ndarray, is_final: bool = False) -> None:
        """Package audio array and push onto queue."""
        now = time.time()
        elapsed = now - self._session_start
        duration = len(audio) / SAMPLE_RATE
        item = {
            "audio":      audio,
            "start_time": elapsed - duration,
            "end_time":   elapsed,
            "is_final":   is_final,
        }
        self.chunk_queue.put(item)


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    capture = AudioCapture()
    capture.start()

    print(f"Recording for {CHUNK_SECONDS * 2 + 5}s — speak into your mic...")
    try:
        while True:
            chunk = capture.get_chunk(timeout=1.0)
            if chunk:
                duration = chunk["end_time"] - chunk["start_time"]
                rms = float(np.sqrt(np.mean(chunk["audio"] ** 2)))
                print(f"  chunk  t={chunk['start_time']:.1f}s–{chunk['end_time']:.1f}s  "
                      f"dur={duration:.1f}s  rms={rms:.4f}  final={chunk['is_final']}")
    except KeyboardInterrupt:
        pass

    capture.stop()
    print("Done.")
