"""
speaker_registry.py
--------------------
Resolves pyannote's per-chunk speaker IDs (SPEAKER_00, SPEAKER_01, ...)
into consistent session-level labels, merging known names with anonymous ones.

Problem:
  pyannote assigns IDs independently per chunk, so SPEAKER_00 in chunk 1
  may be a different person than SPEAKER_00 in chunk 2.

Solution:
  We maintain an embedding fingerprint per speaker (using pyannote's
  embedding model) and do cosine similarity matching across chunks.
  Known speakers get real names; unknown speakers get "Speaker 3" etc.

Pre-registered speakers:
  At session start, call registry.register("Aditya") etc. for known
  attendees. Their names will appear once diarization assigns them.
  Unknown speakers are auto-labelled and can be renamed at any time.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from pyannote.audio import Model, Inference
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

from config import HF_TOKEN, SAMPLE_RATE


COSINE_THRESHOLD = 0.75   # similarity above this → same speaker


@dataclass
class SpeakerProfile:
    label: str                      # display name: "Aditya" or "Speaker 2"
    is_known: bool                  # pre-registered vs auto-assigned
    embedding: Optional[np.ndarray] = field(default=None, repr=False)
    chunk_ids: list[str]            = field(default_factory=list)
                                    # pyannote IDs seen for this speaker


class SpeakerRegistry:
    """
    Session-scoped speaker identity resolver.
    Create one instance per session and keep it alive.
    """

    def __init__(self):
        self._profiles: list[SpeakerProfile] = []
        self._next_anon_index = 1
        self._inference: Optional["Inference"] = None   # embedding model

    # ── Setup ──────────────────────────────────────────────────────────────

    def load_embedding_model(self) -> None:
        """
        Load pyannote's speaker embedding model.
        Optional but strongly recommended — without it, cross-chunk speaker
        consistency falls back to a simpler heuristic.
        """
        if not EMBEDDING_AVAILABLE:
            print("[SpeakerRegistry] pyannote not available — using heuristic mode")
            return
        if not HF_TOKEN:
            print("[SpeakerRegistry] no HF_TOKEN — skipping embedding model")
            return

        model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=HF_TOKEN,
        )
        self._inference = Inference(model, window="whole")
        print("[SpeakerRegistry] embedding model loaded")

    def register(self, name: str) -> None:
        """Pre-register a known attendee by name."""
        profile = SpeakerProfile(label=name, is_known=True)
        self._profiles.append(profile)
        print(f"[SpeakerRegistry] registered known speaker: '{name}'")

    def rename(self, current_label: str, new_label: str) -> bool:
        """Rename a speaker (e.g. 'Speaker 2' → 'Rohan') at any time."""
        for p in self._profiles:
            if p.label == current_label:
                p.label = new_label
                p.is_known = True
                print(f"[SpeakerRegistry] renamed '{current_label}' → '{new_label}'")
                return True
        return False

    def get_all_speakers(self) -> list[dict]:
        """Return all known speaker profiles for the UI."""
        return [
            {"label": p.label, "is_known": p.is_known, "chunk_ids": p.chunk_ids}
            for p in self._profiles
        ]

    # ── Resolution ─────────────────────────────────────────────────────────

    def resolve(
        self,
        chunk_speaker_id: str,
        audio_segment: Optional[np.ndarray] = None,
    ) -> str:
        """
        Map a pyannote chunk-level speaker ID to a session-level label.

        Args:
            chunk_speaker_id:  e.g. "SPEAKER_00" from this chunk's diarization
            audio_segment:     the raw audio for this speaker's turn (optional,
                               used for embedding-based matching)

        Returns:
            Display label, e.g. "Aditya", "Speaker 2"
        """
        # If we have embeddings, try similarity matching first
        if audio_segment is not None and self._inference is not None:
            embedding = self._compute_embedding(audio_segment)
            if embedding is not None:
                match = self._find_by_embedding(embedding)
                if match:
                    if chunk_speaker_id not in match.chunk_ids:
                        match.chunk_ids.append(chunk_speaker_id)
                    if embedding is not None:
                        match.embedding = self._update_embedding(match.embedding, embedding)
                    return match.label

        # Fall back: check if this chunk ID was already mapped in this chunk
        for p in self._profiles:
            if chunk_speaker_id in p.chunk_ids:
                return p.label

        # New speaker — assign to the next unmatched known profile or create anon
        label = self._assign_new(chunk_speaker_id, audio_segment)
        return label

    def resolve_segments(
        self,
        segments: list[dict],
        audio: Optional[np.ndarray] = None,
        diarization_raw: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Resolve speaker labels for a full list of transcript segments.
        Replaces the 'speaker' field (SPEAKER_00 etc.) with display labels.
        """
        for seg in segments:
            raw_id = seg.get("speaker", "SPEAKER_UNKNOWN")

            # Extract the audio slice for this segment if available
            audio_slice = None
            if audio is not None:
                start_sample = int(seg["start"] * SAMPLE_RATE)
                end_sample   = int(seg["end"]   * SAMPLE_RATE)
                audio_slice  = audio[start_sample:end_sample]
                if len(audio_slice) < SAMPLE_RATE * 0.5:
                    audio_slice = None   # too short for reliable embedding

            seg["speaker"] = self.resolve(raw_id, audio_slice)

        return segments

    # ── Internal ───────────────────────────────────────────────────────────

    def _assign_new(self, chunk_id: str, audio: Optional[np.ndarray]) -> str:
        """Create a new speaker profile and return its label."""
        embedding = None
        if audio is not None and self._inference is not None:
            embedding = self._compute_embedding(audio)

        # Try to match to an unmatched known (pre-registered) speaker first
        for p in self._profiles:
            if p.is_known and not p.chunk_ids:
                p.chunk_ids.append(chunk_id)
                p.embedding = embedding
                return p.label

        # Create anonymous speaker
        label = f"Speaker {self._next_anon_index}"
        self._next_anon_index += 1
        profile = SpeakerProfile(
            label=label,
            is_known=False,
            embedding=embedding,
            chunk_ids=[chunk_id],
        )
        self._profiles.append(profile)
        return label

    def _compute_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        try:
            import torch
            waveform = torch.tensor(audio).unsqueeze(0)
            result   = self._inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})
            return np.array(result)
        except Exception as e:
            print(f"[SpeakerRegistry] embedding error: {e}")
            return None

    def _find_by_embedding(self, embedding: np.ndarray) -> Optional[SpeakerProfile]:
        """Return the best-matching profile above COSINE_THRESHOLD, or None."""
        best_profile = None
        best_sim     = COSINE_THRESHOLD

        for p in self._profiles:
            if p.embedding is None:
                continue
            sim = self._cosine(embedding, p.embedding)
            if sim > best_sim:
                best_sim    = sim
                best_profile = p

        return best_profile

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0

    @staticmethod
    def _update_embedding(old: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        """Exponential moving average to keep the embedding up to date."""
        if old is None:
            return new
        return 0.85 * old + 0.15 * new
