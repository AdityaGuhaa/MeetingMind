"""
diarizer.py
-----------
Wraps pyannote.audio to assign speaker labels to audio chunks.

pyannote outputs a timeline of (start, end, speaker_id) tuples.
We then align these with Whisper's word-level timestamps to tag
each transcript segment with the speaker who was talking.

Important: pyannote works best on the full audio. In near-live mode
we run it per-chunk, which means speaker IDs (SPEAKER_00, SPEAKER_01)
may not be consistent across chunks. speaker_registry.py handles
the cross-chunk ID normalisation.
"""

import numpy as np
import torch

from pyannote.audio import Pipeline

from config import HF_TOKEN, DIARIZE_MODEL, SAMPLE_RATE, MAX_SPEAKERS


class Diarizer:
    """Assigns speaker labels to a chunk of audio."""

    def __init__(self):
        self._pipeline: Pipeline | None = None

    def load(self) -> None:
        """
        Load the pyannote diarization pipeline.
        Requires HF_TOKEN with access to pyannote/speaker-diarization-3.1.

        Steps to get access (one-time):
          1. Create account at https://huggingface.co
          2. Visit https://hf.co/pyannote/speaker-diarization-3.1 → accept terms
          3. Visit https://hf.co/pyannote/segmentation-3.0 → accept terms
          4. Generate token at https://hf.co/settings/tokens
          5. Add HF_TOKEN=<your_token> to .env
        """
        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is empty. Add it to your .env file.\n"
                "See the docstring in diarizer.py for setup instructions."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Diarizer] loading '{DIARIZE_MODEL}' on {device}")

        self._pipeline = Pipeline.from_pretrained(
            DIARIZE_MODEL,
            use_auth_token=HF_TOKEN,
        )
        self._pipeline.to(device)
        print("[Diarizer] pipeline ready")

    def diarize(self, audio: np.ndarray) -> list[dict]:
        """
        Run diarization on a float32 16 kHz mono numpy array.

        Returns a list of speaker segments:
            [
                {"start": 0.0, "end": 3.2, "speaker": "SPEAKER_00"},
                {"start": 3.2, "end": 7.1, "speaker": "SPEAKER_01"},
                ...
            ]
        """
        if self._pipeline is None:
            raise RuntimeError("Diarizer.load() must be called first")

        # pyannote expects a dict with "waveform" (tensor) and "sample_rate"
        waveform = torch.tensor(audio).unsqueeze(0)   # shape: (1, N)
        input_data = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        diarization = self._pipeline(
            input_data,
            max_speakers=MAX_SPEAKERS,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start":   round(turn.start, 3),
                "end":     round(turn.end,   3),
                "speaker": speaker,             # e.g. "SPEAKER_00"
            })

        return segments

    def assign_speakers(
        self,
        transcript_segments: list[dict],
        diarization_segments: list[dict],
        chunk_start_time: float = 0.0,
    ) -> list[dict]:
        """
        Merge Whisper transcript segments with pyannote speaker segments.

        Strategy: for each transcript segment, find the diarization segment
        with the maximum time overlap and assign that speaker label.
        Ties go to the earlier speaker (first to speak wins the segment).

        Args:
            transcript_segments:  output of Transcriber.transcribe()
            diarization_segments: output of Diarizer.diarize()
            chunk_start_time:     added to diarization timestamps to bring
                                  them into session-absolute time

        Returns:
            transcript_segments with a "speaker" key added to each dict.
        """
        # Shift diarization timestamps to session-absolute time
        shifted = [
            {
                "start":   d["start"] + chunk_start_time,
                "end":     d["end"]   + chunk_start_time,
                "speaker": d["speaker"],
            }
            for d in diarization_segments
        ]

        for seg in transcript_segments:
            seg_start = seg["start"]
            seg_end   = seg["end"]
            best_speaker = "SPEAKER_UNKNOWN"
            best_overlap = 0.0

            for d in shifted:
                overlap = min(seg_end, d["end"]) - max(seg_start, d["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d["speaker"]

            seg["speaker"] = best_speaker

        return transcript_segments


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import soundfile as sf

    dz = Diarizer()
    dz.load()

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]

        print(f"Diarizing {len(audio)/sr:.1f}s of audio...")
        segs = dz.diarize(audio)
        for s in segs:
            print(f"  [{s['start']:.2f}s → {s['end']:.2f}s]  {s['speaker']}")
    else:
        print("Usage: python diarizer.py path/to/audio.wav")
