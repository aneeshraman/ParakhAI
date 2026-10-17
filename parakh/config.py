"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or default


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str = field(default_factory=lambda: _env("SARVAM_API_KEY"))

    # Sarvam Vision (Document Intelligence). Leave the model unset to use Sarvam's default.
    ocr_model: str | None = field(default_factory=lambda: _env("SARVAM_OCR_MODEL") or None)
    ocr_poll_seconds: float = field(default_factory=lambda: float(_env("SARVAM_OCR_POLL_SECONDS", "3")))
    ocr_timeout_seconds: float = field(default_factory=lambda: float(_env("SARVAM_OCR_TIMEOUT_SECONDS", "600")))

    # "sarvam-translate:v1" covers all 22 scheduled languages (2000 chars per request);
    # "mayura:v1" covers 11 languages (1000 chars per request) but supports colloquial modes.
    translate_model: str = field(default_factory=lambda: _env("SARVAM_TRANSLATE_MODEL", "sarvam-translate:v1"))

    # Translation checks beyond the free deterministic ones (see parakh/checks.py).
    # Back-translation roughly doubles translation cost; the judge is one LLM call per ~12K chars.
    check_back_translation: bool = field(default_factory=lambda: _env("CHECK_BACK_TRANSLATION", "true").lower() == "true")
    check_judge: bool = field(default_factory=lambda: _env("CHECK_JUDGE", "true").lower() == "true")

    # Final reasoning LLM.
    #   gemini            -> Google AI Studio (free key), serves Gemma 4 natively
    #   openai_compatible -> any /chat/completions server: OpenRouter, Ollama, vLLM, Groq...
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "gemini"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemma-4-31b-it"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY") or _env("GEMINI_API_KEY"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://openrouter.ai/api/v1"))
    llm_max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "8192")))
    llm_timeout_seconds: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "300")))

    def require_sarvam_key(self) -> str:
        if not self.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not set. Copy .env.example to .env and add your key.")
        return self.sarvam_api_key
