"""Context compression — reduces retrieved chunks to keep only relevant content.

Uses LLMLingua if available, otherwise falls back to smart extractive compression.
"""
import logging
import re

from src.core.models import Chunk

logger = logging.getLogger(__name__)

# ── LLMLingua (optional, lazy-loaded) ─────────────────────────────────────────

_compressor = None
_LLMLINGUA_AVAILABLE = False


def _get_compressor():
    """Lazy-load LLMLingua to avoid import-time crashes."""
    global _compressor, _LLMLINGUA_AVAILABLE
    if _compressor is not None:
        return _compressor
    try:
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        )
        _LLMLINGUA_AVAILABLE = True
        logger.info("LLMLingua compressor loaded successfully")
        return _compressor
    except Exception as e:
        logger.info(f"LLMLingua not available ({e}) — using extractive compression")
        _LLMLINGUA_AVAILABLE = False
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def compress_chunks(chunks: list[Chunk], query: str, target_ratio: float = 0.8) -> str:
    """Compress retrieved chunks to reduce token usage while preserving relevance."""
    if not chunks:
        return ""

    full_context = _build_context(chunks)

    if not full_context.strip():
        return ""

    # Skip compression for small contexts (not worth it)
    if len(full_context) < 2000:
        logger.info(f"Context small enough ({len(full_context)} chars) — skipping compression")
        return full_context

    # Try LLMLingua first
    compressor = _get_compressor()
    if compressor is not None:
        try:
            result = compressor.compress_prompt(
                full_context,
                instruction=query,
                target_token=max(50, int(len(full_context.split()) * target_ratio)),
            )
            compressed = result.get("compressed_prompt", "")
            if compressed and len(compressed.strip()) > 20:
                logger.info(
                    f"LLMLingua compressed: {len(full_context)} → {len(compressed)} chars "
                    f"({len(compressed)/max(1,len(full_context))*100:.0f}%)"
                )
                return compressed
        except Exception as e:
            logger.warning(f"LLMLingua compression failed: {e} — falling back to extractive")

    # Extractive fallback
    result = _extractive_compress(chunks, query)
    logger.info(f"Extractive compressed: {len(full_context)} → {len(result)} chars")
    return result


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_context(chunks: list[Chunk]) -> str:
    """Build a formatted context string from chunks."""
    parts = []
    for c in chunks:
        speaker = f"[{c.speaker}] " if c.speaker else ""
        ts = f"({c.start_time:.0f}s) " if c.start_time is not None and c.start_time > 0 else ""
        parts.append(f"{speaker}{ts}{c.text}")
    return "\n\n".join(parts)


def _extractive_compress(chunks: list[Chunk], query: str) -> str:
    """Smart extractive compression — keeps sentences relevant to the query.

    Strategy:
    1. Extract keywords from query (words > 3 chars, lowercase)
    2. For each chunk, score each sentence by keyword overlap
    3. Keep sentences that match any keyword, plus first/last for context
    4. If nothing matches, keep the full chunk (don't lose data)
    """
    # Extract meaningful keywords (skip short/common words)
    stop_words = {"what", "where", "when", "this", "that", "with", "from", "have",
                  "were", "they", "their", "about", "which", "there", "been", "will",
                  "would", "could", "should", "does", "made", "also", "some", "many"}
    keywords = {
        w.lower()
        for w in re.findall(r'\b[a-zA-Z]+\b', query)
        if len(w) > 3 and w.lower() not in stop_words
    }

    parts = []
    for c in chunks:
        # Split into sentences
        sentences = _split_sentences(c.text)
        if not sentences:
            continue

        if not keywords:
            # No usable keywords — keep everything
            kept = sentences
        else:
            # Score each sentence
            scored = []
            for s in sentences:
                s_lower = s.lower()
                score = sum(1 for kw in keywords if kw in s_lower)
                scored.append((s, score))

            # Keep all sentences with at least 1 keyword match
            kept = [s for s, score in scored if score > 0]

            if not kept:
                # No matches — keep first 3 sentences as context
                kept = sentences[:3]
            elif len(kept) < len(sentences):
                # Add first sentence for context if not already included
                if sentences[0] not in kept:
                    kept.insert(0, sentences[0])

        speaker = f"[{c.speaker}] " if c.speaker else ""
        ts = f"({c.start_time:.0f}s) " if c.start_time is not None and c.start_time > 0 else ""
        parts.append(f"{speaker}{ts}" + " ".join(kept))

    return "\n\n".join(parts)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common edge cases."""
    # Split on sentence endings followed by space/newline
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 5]
