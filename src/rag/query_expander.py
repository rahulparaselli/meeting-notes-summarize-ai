"""HyDE (Hypothetical Document Embeddings) query expansion.

Instead of searching with the raw query, generates a hypothetical transcript
excerpt that would answer the question. This produces embeddings that are
closer to actual document embeddings, improving retrieval quality.
"""
import logging

from src.core.llm import generate_text

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = """You are a meeting transcript generator. Given a question about a meeting,
write a SHORT, realistic excerpt (2-3 sentences) from a meeting transcript that
would contain the answer to this question.

Rules:
- Write as if you're quoting a real meeting transcript
- Include speaker names like "Alice:", "Bob:", etc.
- Be specific and factual-sounding
- Do NOT answer the question — just write what the transcript might look like
- Keep it under 50 words"""


async def expand_query_hyde(query: str) -> str:
    """Generate a hypothetical transcript passage for better embedding retrieval.

    Returns the original query combined with a hypothetical answer passage.
    If HyDE generation fails, returns the original query unchanged.
    """
    try:
        hypothetical = await generate_text(
            prompt=f"Question about a meeting: {query}\n\nWrite a short hypothetical meeting transcript excerpt that would answer this:",
            system=_HYDE_SYSTEM,
            temperature=0.3,
        )

        if hypothetical and len(hypothetical.strip()) > 10:
            combined = f"{query}\n\n{hypothetical.strip()}"
            logger.info(f"HyDE expanded query: '{query[:50]}' → {len(combined)} chars")
            return combined
        else:
            logger.warning("HyDE returned empty/short response, using original query")
            return query

    except Exception as e:
        logger.warning(f"HyDE expansion failed: {e} — using original query")
        return query


async def expand_multi_query(query: str, n: int = 3) -> list[str]:
    """Rephrase query in N ways to improve recall."""
    try:
        prompt = (
            f"Rephrase this meeting query in {n} different ways. "
            f"Each rephrasing should focus on a different aspect.\n"
            f"Return each on a new line, no numbering.\n\nQuery: {query}"
        )
        result = await generate_text(prompt=prompt, temperature=0.4)
        variants = [q.strip() for q in result.splitlines() if q.strip() and len(q.strip()) > 5]
        return [query] + variants[:n]
    except Exception as e:
        logger.warning(f"Multi-query expansion failed: {e}")
        return [query]
