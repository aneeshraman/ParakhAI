"""Stage 3: hand the text to an open-weight LLM (Gemma by default)."""

from __future__ import annotations

from typing import Protocol

import httpx
from google import genai
from google.genai import errors, types

from .config import Settings


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...


class GeminiAPILLM:
    """Gemma served by Google AI Studio. A free API key from aistudio.google.com is enough."""

    def __init__(self, settings: Settings):
        if not settings.llm_api_key:
            raise LLMError("Set GEMINI_API_KEY (free at https://aistudio.google.com/apikey).")
        self.name = f"gemini:{settings.llm_model}"
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.client = genai.Client(
            api_key=settings.llm_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.llm_timeout_seconds * 1000),  # milliseconds
                # The free tier returns 429s under bursty load and 500/503s when Google is busy;
                # back off (2s, 4s, ... up to 60s) for about 4 minutes before giving up.
                retry_options=types.HttpRetryOptions(
                    attempts=8, initial_delay=2.0, max_delay=60.0, http_status_codes=[429, 500, 502, 503, 504]
                ),
            ),
        )

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=self.max_tokens),
            )
        except errors.APIError as e:
            raise LLMError(f"{self.name} failed with HTTP {e.code}: {e.message}") from e
        if not response.text:
            raise LLMError(f"{self.name} returned no text (finish reason: {_finish_reason(response)}).")
        return response.text


class OpenAICompatibleLLM:
    """Any server exposing POST {base_url}/chat/completions, e.g. OpenRouter
    (model 'google/gemma-4-31b-it:free'), or Ollama/vLLM running Gemma locally."""

    def __init__(self, settings: Settings):
        self.name = f"openai_compatible:{settings.llm_model}"
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
        self.http = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            headers=headers,
            timeout=settings.llm_timeout_seconds,
        )

    def generate(self, system: str, prompt: str) -> str:
        try:
            response = self.http.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name} request failed: {e}") from e
        if response.status_code >= 400:
            raise LLMError(f"{self.name} returned HTTP {response.status_code}: {response.text[:500]}")
        content = response.json()["choices"][0]["message"].get("content")
        if not content:
            raise LLMError(f"{self.name} returned no text.")
        return content


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    return str(candidates[0].finish_reason) if candidates else "unknown"


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "gemini":
        return GeminiAPILLM(settings)
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(settings)
    raise LLMError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use 'gemini' or 'openai_compatible'.")
