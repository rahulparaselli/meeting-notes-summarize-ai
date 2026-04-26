from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.core.config import get_settings

_settings = get_settings()

# singleton async client
_client = genai.Client(api_key=_settings.gemini_api_key)

T = TypeVar("T", bound=BaseModel)


async def generate_structured(
    prompt: str,
    schema: type[T],
    system: str = "",
    temperature: float | None = None,
) -> T:
    """Generate with native Gemini response_schema (zero parse failures)."""
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature or _settings.gemma_temperature,
        max_output_tokens=_settings.gemma_max_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )
    response = await _client.aio.models.generate_content(
        model=_settings.gemma_model,
        contents=prompt,
        config=config,
    )
    return schema.model_validate_json(response.text)


async def generate_text(
    prompt: str,
    system: str = "",
    temperature: float | None = None,
) -> str:
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature or _settings.gemma_temperature,
        max_output_tokens=_settings.gemma_max_tokens,
    )
    response = await _client.aio.models.generate_content(
        model=_settings.gemma_model,
        contents=prompt,
        config=config,
    )
    return response.text.strip()


async def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    result = await _client.aio.models.embed_content(
        model=_settings.embedding_model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return result.embeddings[0].values  # type: ignore[index]


async def embed_batch(
    texts: list[str], task_type: str = "retrieval_document"
) -> list[list[float]]:
    result = await _client.aio.models.embed_content(
        model=_settings.embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in result.embeddings]  # type: ignore[union-attr]
