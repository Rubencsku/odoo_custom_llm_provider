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


class OllamaAdapter(BaseAIAdapter):
    """
    Adapter for native Ollama API (e.g. http://localhost:11434 or LAN host).
    Supports automatic local model detection via /api/tags and private local inference.
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
        endpoint = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(
                endpoint,
                headers=self._get_headers(),
                timeout=min(self.timeout, 10),
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                return True, f"Ollama local connection successful! Found {len(models)} local models installed."
            else:
                return False, f"Ollama server returned HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.exceptions.Timeout:
            return False, f"Connection timed out reaching Ollama at {self.base_url}"
        except Exception as e:
            return False, f"Cannot connect to Ollama: {str(e)}"

    def fetch_available_models(self) -> List[ModelInfo]:
        endpoint = f"{self.base_url}/api/tags"
        try:
            resp = requests.get(endpoint, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code != 200:
                raise self._normalize_error(resp.status_code, resp.text)
            data = resp.json()
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Error fetching Ollama models: {re}")

        models = []
        for item in data.get("models", []):
            name = item.get("name") or item.get("model")
            if not name:
                continue

            lower = name.lower()
            is_vision = any(v in lower for v in ["llava", "vision", "vl", "minicpm"])
            is_embed = "embed" in lower or "bge" in lower or "nomic" in lower

            models.append(
                ModelInfo(
                    model_id=name,
                    name=f"Ollama {name}",
                    capability_chat=not is_embed,
                    capability_vision=is_vision,
                    capability_tools=True,
                    capability_json=True,
                    capability_embeddings=is_embed,
                    description=f"Local Ollama model: {name}",
                )
            )
        return models

    def get_embedding(
        self, input, dimensions: Optional[int], model: str,
        encoding_format: Optional[str] = None, user: Optional[str] = None,
    ) -> Dict[str, Any]:
        endpoint = f"{self.base_url}/api/embed"
        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json={"model": model, "input": input},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise self._normalize_error(response.status_code, response.text)
            payload = response.json()
        except requests.exceptions.Timeout as error:
            raise AITimeoutError(f"Ollama embedding request timed out: {error}")
        except requests.exceptions.RequestException as error:
            raise AIProviderError(f"Ollama embedding request failed: {error}")

        vectors = payload.get("embeddings") or []
        result = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": index, "embedding": vector}
                for index, vector in enumerate(vectors)
            ],
            "model": model,
            "usage": {
                "prompt_tokens": payload.get("prompt_eval_count", 0),
                "total_tokens": payload.get("prompt_eval_count", 0),
            },
        }
        return self._fit_embedding_dimensions(result, dimensions)

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
        endpoint = f"{self.base_url}/api/chat"
        headers = self._get_headers()

        options: Dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if top_p is not None:
            options["top_p"] = top_p

        normalized_messages = []
        for message in messages:
            normalized = dict(message)
            if isinstance(normalized.get("content"), list):
                texts = []
                images = []
                for part in normalized["content"]:
                    if part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if ";base64," in image_url:
                            images.append(image_url.split(",", 1)[1])
                normalized["content"] = "\n".join(texts)
                if images:
                    normalized["images"] = images
            normalized_messages.append(normalized)

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": normalized_messages,
            "stream": False,
        }

        if options:
            payload["options"] = options
        if response_format and response_format.get("type") == "json_object":
            payload["format"] = "json"
        elif response_format and response_format.get("type") == "json_schema":
            payload["format"] = response_format["json_schema"]["schema"]
        if tools:
            payload["tools"] = tools

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
            raise AITimeoutError(f"Ollama inference timed out after {self.timeout}s: {te}")
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Ollama request error: {re}")

        message = res_json.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls", [])

        prompt_tokens = res_json.get("prompt_eval_count", 0)
        completion_tokens = res_json.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Ollama gives total_duration in nanoseconds
        if "total_duration" in res_json and res_json["total_duration"]:
            latency_ms = res_json["total_duration"] / 1_000_000.0

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason="stop",
            raw=res_json,
        )
