from src.core.models import Chunk

try:
    from llmlingua import PromptCompressor  # type: ignore[import-untyped]
    _compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
    _LLMLINGUA_AVAILABLE = True
except (ImportError, Exception):
    _LLMLINGUA_AVAILABLE = False


def compress_chunks(chunks: list[Chunk], query: str, target_ratio: float = 0.6) -> str:
    """Compress retrieved chunks to reduce token usage."""
    full_context = _build_context(chunks)

    if _LLMLINGUA_AVAILABLE:
        try:
            result = _compressor.compress_prompt(  # type: ignore[union-attr]
                full_context,
                instruction=query,
                target_token=int(len(full_context.split()) * target_ratio),
            )
            return result["compressed_prompt"]
        except Exception:
            pass  # fall through to extractive

    return _extractive_compress(chunks, query)


def _build_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        speaker = f"[{c.speaker}] " if c.speaker else ""
        ts = f"({c.start_time:.0f}s)" if c.start_time is not None else ""
        parts.append(f"{speaker}{ts} {c.text}")
    return "\n\n".join(parts)


def _extractive_compress(chunks: list[Chunk], query: str) -> str:
    """Simple extractive: keep sentences containing query keywords."""
    keywords = {w.lower() for w in query.split() if len(w) > 3}
    parts = []
    for c in chunks:
        sentences = [s for s in c.text.split(". ") if s]
        kept = [
            s for s in sentences
            if any(kw in s.lower() for kw in keywords)
        ] or sentences[:2]  # fallback: keep first 2
        speaker = f"[{c.speaker}] " if c.speaker else ""
        ts = f"({c.start_time:.0f}s) " if c.start_time is not None else ""
        parts.append(f"{speaker}{ts}" + ". ".join(kept))
    return "\n\n".join(parts)
