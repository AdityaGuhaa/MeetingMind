"""
rag.py
------
Post-session retrieval-augmented generation (RAG) using ChromaDB.

Pipeline:
  1. Chunk the full transcript into overlapping text segments
  2. Embed each chunk using nomic-embed-text via Ollama
  3. Store in ChromaDB (persisted to disk per session)
  4. At query time: embed the question, retrieve top-K chunks, pass to LLM

Requires:
  ollama pull nomic-embed-text
"""

import json
import time
import httpx
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DIR, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL
)


class RAGIndex:
    """Manages the ChromaDB vector index for a single session."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._client    = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # Each session gets its own ChromaDB collection
        self._collection = self._client.get_or_create_collection(
            name=f"session_{session_id}",
            metadata={"hnsw:space": "cosine"},
        )
        self._http = httpx.Client(timeout=60.0)
        print(f"[RAG] index ready for session '{session_id}'")

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_transcript(self, transcript: list[dict]) -> int:
        """
        Chunk and index the full session transcript.

        Args:
            transcript: list of segment dicts from SessionManager

        Returns:
            Number of chunks indexed.
        """
        full_text = self._transcript_to_text(transcript)
        chunks    = self._chunk_text(full_text)

        if not chunks:
            return 0

        print(f"[RAG] embedding {len(chunks)} chunks...")
        embeddings = [self._embed(c) for c in chunks]

        # Upsert into ChromaDB (safe to call multiple times)
        self._collection.upsert(
            ids        = [f"{self.session_id}_chunk_{i}" for i in range(len(chunks))],
            documents  = chunks,
            embeddings = embeddings,
        )
        print(f"[RAG] indexed {len(chunks)} chunks")
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = 5) -> str:
        """
        Retrieve the most relevant transcript chunks for a question.

        Returns a single formatted string of the top-K chunks,
        ready to be inserted into an LLM prompt as context.
        """
        if self._collection.count() == 0:
            return "[No transcript indexed yet. Run post-session analysis first.]"

        q_embedding = self._embed(question)
        results     = self._collection.query(
            query_embeddings=[q_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        chunks = results.get("documents", [[]])[0]
        if not chunks:
            return "[No relevant transcript sections found.]"

        # Format with separator so the LLM can distinguish chunks
        formatted = "\n\n---\n\n".join(chunks)
        return formatted

    # ── Helpers ───────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Call Ollama's embedding endpoint for a single text string."""
        response = self._http.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    @staticmethod
    def _transcript_to_text(transcript: list[dict]) -> str:
        """Convert transcript segments to a plain text string for chunking."""
        lines = []
        for seg in transcript:
            spk  = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            t    = seg.get("start", 0)
            mins, secs = divmod(int(t), 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {spk}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """
        Sliding window chunker with character-level overlap.
        Splits on sentence boundaries when possible.
        """
        if not text:
            return []

        chunks = []
        start  = 0
        length = len(text)

        while start < length:
            end = min(start + CHUNK_SIZE, length)

            # Try to break at a newline or sentence boundary
            if end < length:
                for boundary in ("\n", ". ", "? ", "! "):
                    pos = text.rfind(boundary, start, end)
                    if pos > start + CHUNK_SIZE // 2:
                        end = pos + len(boundary)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - CHUNK_OVERLAP

        return chunks
