# -*- coding: utf-8 -*-
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import requests

from .base_adapter import (
    BaseAIAdapter,
    AIResponse,
    ModelInfo,
    AIProviderError,
    AITimeoutError,
)

_logger = logging.getLogger(__name__)


class OpenAICompatibleAdapter(BaseAIAdapter):
    """
    Adapter for OpenAI and all OpenAI-compatible endpoints:
    - Official OpenAI (api.openai.com/v1)
    - OpenRouter (openrouter.ai/api/v1)
    - Groq (api.groq.com/openai/v1)
    - DeepSeek (api.deepseek.com/v1)
    - vLLM / LocalAI / LM Studio / Ollama (/v1)
    """

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.custom_headers:
            headers.update(self.custom_headers)
        return headers

    def test_connection(self) -> Tuple[bool, str]:
        endpoint = f"{self.base_url}/models"
        try:
            resp = requests.get(
                endpoint,
                headers=self._get_headers(),
                timeout=min(self.timeout, 15),
            )
            if resp.status_code == 200:
                data = resp.json()
                count = len(data.get("data", []))
                return True, f"Connection successful! Retrieved {count} models from endpoint."
            else:
                return False, f"Server returned error code {resp.status_code}: {resp.text[:300]}"
        except requests.exceptions.Timeout:
            return False, f"Connection timed out after {min(self.timeout, 15)} seconds connecting to {self.base_url}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def fetch_available_models(self) -> List[ModelInfo]:
        endpoint = f"{self.base_url}/models"
        headers = self._get_headers()
        try:
            resp = requests.get(endpoint, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                raise self._normalize_error(resp.status_code, resp.text)
            data = resp.json()
        except requests.exceptions.Timeout as te:
            raise AITimeoutError(f"Timeout fetching models from {endpoint}: {te}")
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Network error fetching models: {re}")

        raw_models = data.get("data", [])
        models_list: List[ModelInfo] = []

        for item in raw_models:
            model_id = item.get("id") if isinstance(item, dict) else str(item)
            if not model_id:
                continue

            # Skip common non-chat embeddings/audio/whisper/moderation models unless requested
            lower_id = model_id.lower()
            is_embedding = "embed" in lower_id
            is_moderation = "moderation" in lower_id
            is_audio = "whisper" in lower_id or "tts" in lower_id

            if is_moderation:
                continue

            # Heuristics for capabilities
            is_vision = any(k in lower_id for k in ["vision", "vl", "omni", "gpt-4o", "4.1", "gemini", "claude", "qwen-vl"])
            is_chat = not is_embedding and not is_audio

            name = item.get("name") or model_id
            models_list.append(
                ModelInfo(
                    model_id=model_id,
                    name=name,
                    capability_chat=is_chat,
                    capability_vision=is_vision,
                    capability_tools=is_chat,
                    capability_json=is_chat,
                    capability_embeddings=is_embedding,
                    description=item.get("description", "") or f"Model {model_id}",
                )
            )

        return models_list

    def get_embedding(
        self, input, dimensions: Optional[int], model: str,
        encoding_format: Optional[str] = None, user: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {"input": input, "model": model}
        if dimensions:
            payload["dimensions"] = dimensions
        if encoding_format:
            payload["encoding_format"] = encoding_format
        if user:
            payload["user"] = user
        endpoint = f"{self.base_url}/embeddings"
        try:
            response = requests.post(
                endpoint, headers=self._get_headers(), json=payload, timeout=self.timeout
            )
            if response.status_code != 200:
                raise self._normalize_error(response.status_code, response.text)
            return self._fit_embedding_dimensions(response.json(), dimensions)
        except requests.exceptions.Timeout as error:
            raise AITimeoutError(f"Embedding request timed out after {self.timeout}s: {error}")
        except requests.exceptions.RequestException as error:
            raise AIProviderError(f"Embedding request failed: {error}")

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
        endpoint = f"{self.base_url}/chat/completions"
        headers = self._get_headers()

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        # Add any extra vendor specific kwargs
        for k, v in kwargs.items():
            if v is not None and k not in payload:
                payload[k] = v

        start_time = time.time()
        try:
            resp = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            latency_ms = (time.time() - start_time) * 1000.0

            if resp.status_code != 200:
                raise self._normalize_error(resp.status_code, resp.text)

            res_json = resp.json()
        except requests.exceptions.Timeout as te:
            raise AITimeoutError(f"Timeout calling {endpoint} after {self.timeout}s: {te}")
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Network error in chat completion: {re}")

        choices = res_json.get("choices", [])
        if not choices:
            raise AIProviderError("Invalid response: missing 'choices' in API return", details=res_json)

        first_choice = choices[0]
        message_data = first_choice.get("message", {})
        content = message_data.get("content") or ""
        tool_calls = message_data.get("tool_calls") or []
        finish_reason = first_choice.get("finish_reason")

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw=res_json,
        )
