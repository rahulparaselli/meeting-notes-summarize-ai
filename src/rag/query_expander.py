from src.core.llm import generate_text

_SYSTEM = (
    "You are a meeting analyst. Generate a concise hypothetical transcript excerpt "
    "that would answer the question. 2-3 sentences max, no preamble."
)


async def expand_query_hyde(query: str) -> str:
    """Generate a hypothetical answer passage for better embedding retrieval."""
    hypothetical = await generate_text(
        prompt=f"Question: {query}",
        system=_SYSTEM,
        temperature=0.3,
    )
    # combine original + hypothetical for richer embedding
    return f"{query}\n\n{hypothetical}"


async def expand_multi_query(query: str, n: int = 3) -> list[str]:
    """Rephrase query in N ways to improve recall."""
    prompt = (
        f"Rephrase this meeting query in {n} different ways. "
        f"Return each on a new line, no numbering.\n\nQuery: {query}"
    )
    result = await generate_text(prompt=prompt, temperature=0.4)
    variants = [q.strip() for q in result.splitlines() if q.strip()]
    return [query] + variants[:n]
