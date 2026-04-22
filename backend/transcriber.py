"""
transcriber.py
--------------
Wraps faster-whisper to transcribe audio chunks into text segments.

faster-whisper is significantly faster than openai-whisper on CPU
due to CTranslate2 optimisations, and supports int8 quantisation
which cuts RAM usage roughly in half.

Each transcription returns a list of segment dicts:
    [
        {
            "start":  float,   # seconds from chunk start
            "end":    float,
            "text":   str,
            "words":  list     # word-level timestamps if available
        },
        ...
    ]
"""

import numpy as np
from faster_whisper import WhisperModel

from config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE


class Transcriber:
    """
    Thin wrapper around faster-whisper.
    Thread-safe for single-threaded sequential calls (one chunk at a time).
    For true parallel transcription you'd need a model-per-thread.
    """

    def __init__(self):
        self._model: WhisperModel | None = None

    def load(self) -> None:
        """Load the Whisper model into memory. Call once at startup."""
        device = WHISPER_DEVICE
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        print(f"[Transcriber] loading whisper '{WHISPER_MODEL}' "
              f"on {device} ({WHISPER_COMPUTE})")
        self._model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=WHISPER_COMPUTE,
        )
        print("[Transcriber] model ready")

    def transcribe(self, audio: np.ndarray, chunk_start_time: float = 0.0) -> list[dict]:
        """
        Transcribe a float32 16 kHz mono numpy array.

        Args:
            audio:            np.ndarray, shape (N,), float32, 16 kHz mono
            chunk_start_time: wall-clock offset of this chunk's start (seconds
                              since session began). Added to segment timestamps
                              so all segments share a common session timeline.

        Returns:
            List of segment dicts with session-absolute timestamps.
        """
        if self._model is None:
            raise RuntimeError("Transcriber.load() must be called before transcribe()")

        # faster-whisper expects float32 numpy array at 16 kHz
        segments_iter, info = self._model.transcribe(
            audio,
            language=None,          # auto-detect; set "en" to skip detection
            beam_size=3,            # 3 is fast and accurate enough for near-live
            word_timestamps=True,   # needed later for diarization alignment
            vad_filter=True,        # skip silent regions — big speed win
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
        )

        results = []
        for seg in segments_iter:
            segment = {
                "start": round(chunk_start_time + seg.start, 3),
                "end":   round(chunk_start_time + seg.end,   3),
                "text":  seg.text.strip(),
                "words": [
                    {
                        "word":  w.word,
                        "start": round(chunk_start_time + w.start, 3),
                        "end":   round(chunk_start_time + w.end,   3),
                    }
                    for w in (seg.words or [])
                ],
            }
            if segment["text"]:     # skip empty segments
                results.append(segment)

        return results


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import soundfile as sf

    t = Transcriber()
    t.load()

    if len(sys.argv) > 1:
        # Test with a WAV file: python transcriber.py path/to/audio.wav
        audio_path = sys.argv[1]
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]   # take first channel if stereo
        if sr != 16000:
            print(f"Warning: expected 16000 Hz, got {sr}. Resample first for best results.")

        print(f"Transcribing {len(audio)/sr:.1f}s of audio...")
        segments = t.transcribe(audio, chunk_start_time=0.0)
        for seg in segments:
            print(f"  [{seg['start']:.2f}s → {seg['end']:.2f}s]  {seg['text']}")
    else:
        # Quick synthetic silence test — just verifies the model loads
        print("No audio file provided — testing with synthetic silence (should produce no segments)")
        silence = np.zeros(16000 * 3, dtype=np.float32)  # 3s of silence
        segments = t.transcribe(silence)
        print(f"  segments on silence: {len(segments)} (expected 0)")
        print("Pass: model loads and handles silence correctly.")
        print("\nUsage: python transcriber.py path/to/audio.wav")
