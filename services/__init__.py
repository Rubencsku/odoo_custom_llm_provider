# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional

from .base_adapter import (
    BaseAIAdapter,
    AIResponse,
    ModelInfo,
    AIProviderError,
    AIAuthenticationError,
    AIRateLimitError,
    AITimeoutError,
    AIServiceUnavailableError,
)
from .openai_adapter import OpenAICompatibleAdapter
from .anthropic_adapter import AnthropicAdapter
from .gemini_adapter import GeminiAdapter
from .ollama_adapter import OllamaAdapter
from .native_bridge import (
    build_messages,
    custom_record_id,
    format_tools,
    parse_tool_calls,
    response_format_for_schema,
    tool_call_history,
    tool_result_message,
)


def get_adapter(
    provider_type: str,
    base_url: str,
    api_key: Optional[str] = None,
    timeout: int = 60,
    custom_headers: Optional[Dict[str, str]] = None,
) -> BaseAIAdapter:
    """
    Factory function returning the appropriate adapter instance for a given provider type.
    """
    p_type = (provider_type or "").lower()

    if p_type == "anthropic":
        url = base_url or "https://api.anthropic.com/v1"
        return AnthropicAdapter(url, api_key, timeout, custom_headers)

    elif p_type == "gemini":
        url = base_url or "https://generativelanguage.googleapis.com/v1beta"
        return GeminiAdapter(url, api_key, timeout, custom_headers)

    elif p_type == "ollama":
        url = base_url or "http://localhost:11434"
        return OllamaAdapter(url, api_key, timeout, custom_headers)

    elif p_type in ("openai", "openrouter", "groq", "deepseek", "custom_openai", "vllm"):
        default_urls = {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
            "deepseek": "https://api.deepseek.com/v1",
        }
        url = base_url or default_urls.get(p_type, "https://api.openai.com/v1")
        return OpenAICompatibleAdapter(url, api_key, timeout, custom_headers)

    else:
        # Fallback to standard OpenAI compatible
        url = base_url or "https://api.openai.com/v1"
        return OpenAICompatibleAdapter(url, api_key, timeout, custom_headers)


__all__ = [
    "BaseAIAdapter",
    "OpenAICompatibleAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "OllamaAdapter",
    "AIResponse",
    "ModelInfo",
    "AIProviderError",
    "AIAuthenticationError",
    "AIRateLimitError",
    "AITimeoutError",
    "AIServiceUnavailableError",
    "build_messages",
    "custom_record_id",
    "format_tools",
    "parse_tool_calls",
    "response_format_for_schema",
    "tool_call_history",
    "tool_result_message",
    "get_adapter",
]
