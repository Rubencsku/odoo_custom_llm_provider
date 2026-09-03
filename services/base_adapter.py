# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

_logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Exceptions
# -------------------------------------------------------------------------

class AIProviderError(Exception):
    """Base exception for all AI Provider operations."""
    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details or {}


class AIAuthenticationError(AIProviderError):
    """Raised when authentication fails (HTTP 401 / 403)."""
    pass


class AIRateLimitError(AIProviderError):
    """Raised when rate limit or quota is exceeded (HTTP 429). Trigger for fallback."""
    pass


class AITimeoutError(AIProviderError):
    """Raised when the request times out. Trigger for fallback."""
    pass


class AIServiceUnavailableError(AIProviderError):
    """Raised when provider returns 5xx (500, 502, 503, 504). Trigger for fallback."""
    pass


# -------------------------------------------------------------------------
# Data Structures
# -------------------------------------------------------------------------

@dataclass
class AIResponse:
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    model_id: str
    name: str
    capability_chat: bool = True
    capability_vision: bool = False
    capability_tools: bool = False
    capability_json: bool = False
    capability_embeddings: bool = False
    description: str = ""


# -------------------------------------------------------------------------
# Base Adapter
# -------------------------------------------------------------------------

class BaseAIAdapter(ABC):
    """Abstract base class for all LLM provider adapters."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        custom_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout or 60
        self.custom_headers = custom_headers or {}

    @abstractmethod
    def chat_completion(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AIResponse:
        """Send chat messages to the provider and return a normalized AIResponse."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Verify endpoint connectivity and credentials. Returns (success, message)."""
        pass

    @abstractmethod
    def fetch_available_models(self) -> List[ModelInfo]:
        """Query provider endpoint dynamically to discover available models."""
        pass

    def get_embedding(
        self, input, dimensions: Optional[int], model: str,
        encoding_format: Optional[str] = None, user: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return an OpenAI-shaped embedding response for Odoo's RAG engine."""
        raise AIProviderError(
            f"Provider adapter {self.__class__.__name__} does not support embeddings"
        )

    @staticmethod
    def _fit_embedding_dimensions(response: Dict[str, Any], dimensions: Optional[int]):
        """Fit provider vectors to Odoo's fixed pgvector size.

        Padding with zeroes preserves cosine similarity for shorter vectors. A
        provider that returns larger vectors is truncated so PostgreSQL can
        store them in Odoo's fixed-size vector column.
        """
        if not dimensions:
            return response
        for item in response.get("data", []):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                continue
            item["embedding"] = (vector[:dimensions] + [0.0] * dimensions)[:dimensions]
        return response

    def _normalize_error(self, status_code: int, error_text: str) -> AIProviderError:
        """Convert HTTP status codes to specific AIProviderError sub-classes."""
        msg = f"HTTP {status_code}: {error_text}"
        if status_code in (401, 403):
            return AIAuthenticationError(f"Authentication failed: {msg}", status_code=status_code)
        elif status_code == 429:
            return AIRateLimitError(f"Rate limit or quota exceeded: {msg}", status_code=status_code)
        elif status_code in (500, 502, 503, 504):
            return AIServiceUnavailableError(f"Provider service unavailable: {msg}", status_code=status_code)
        return AIProviderError(msg, status_code=status_code)
