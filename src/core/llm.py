import asyncio
import logging
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.core.config import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# Validate API key on startup
if _settings.gemini_api_key in ("placeholder", "your_gemini_api_key_here", ""):
    logger.warning(
        "⚠️  GEMINI_API_KEY is not set! AI features will NOT work. "
        "Get a free key from https://aistudio.google.com/apikey and set it in .env"
    )

_client = genai.Client(api_key=_settings.gemini_api_key)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 3
RETRY_DELAYS = [5, 10, 20]  # seconds


async def _retry_on_rate_limit(func, *args, **kwargs):
    """Retry API calls with exponential backoff on 429 errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"Rate limited, retrying in {delay}s (attempt {attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error("Rate limit exceeded after all retries")
                    raise RuntimeError(
                        "Gemini API rate limit exceeded. Please wait a minute and try again. "
                        "Free tier has limited requests per minute/day."
                    ) from e
            else:
                raise


async def generate_structured(
    prompt: str,
    schema: type[T],
    system: str = "",
    temperature: float | None = None,
) -> T:
    """Generate with native Gemini response_schema."""
    config = types.GenerateContentConfig(
        system_instruction=system or None,
        temperature=temperature or _settings.gemma_temperature,
        max_output_tokens=_settings.gemma_max_tokens,
        response_mime_type="application/json",
        response_schema=schema,
    )

    async def _call():
        return await _client.aio.models.generate_content(
            model=_settings.gemma_model,
            contents=prompt,
            config=config,
        )

    try:
        response = await _retry_on_rate_limit(_call)
        return schema.model_validate_json(response.text)
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise


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

    async def _call():
        return await _client.aio.models.generate_content(
            model=_settings.gemma_model,
            contents=prompt,
            config=config,
        )

    try:
        response = await _retry_on_rate_limit(_call)
        return response.text.strip()
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise


async def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    async def _call():
        return await _client.aio.models.embed_content(
            model=_settings.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )

    try:
        result = await _retry_on_rate_limit(_call)
        return result.embeddings[0].values  # type: ignore[index]
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Gemini embed call failed: {e}")
        raise


async def embed_batch(
    texts: list[str], task_type: str = "retrieval_document"
) -> list[list[float]]:
    """Embed multiple texts — automatically batches in groups of 100 (Gemini limit)."""
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]

        async def _call(batch_texts=batch):
            return await _client.aio.models.embed_content(
                model=_settings.embedding_model,
                contents=batch_texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )

        try:
            result = await _retry_on_rate_limit(_call)
            all_embeddings.extend([e.values for e in result.embeddings])  # type: ignore[union-attr]
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Gemini embed_batch call failed for batch {i//BATCH_SIZE}: {e}")
            raise

    return all_embeddings

