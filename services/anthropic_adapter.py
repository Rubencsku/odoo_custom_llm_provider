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


class AnthropicAdapter(BaseAIAdapter):
    """
    Adapter for Anthropic Claude API (https://api.anthropic.com/v1).
    Supports Messages API and native dynamic model discovery via /v1/models.
    """

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
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
                models = data.get("data", [])
                return True, f"Anthropic connection successful! Discovered {len(models)} Claude models."
            elif resp.status_code in (401, 403):
                return False, f"Authentication failed (HTTP {resp.status_code}): Invalid Anthropic API Key."
            else:
                return False, f"Server returned code {resp.status_code}: {resp.text[:300]}"
        except requests.exceptions.Timeout:
            return False, f"Connection timed out connecting to {self.base_url}"
        except Exception as e:
            return False, f"Anthropic connection error: {str(e)}"

    def fetch_available_models(self) -> List[ModelInfo]:
        endpoint = f"{self.base_url}/models"
        try:
            resp = requests.get(endpoint, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                models = []
                for item in data:
                    m_id = item.get("id")
                    name = item.get("display_name") or m_id
                    models.append(
                        ModelInfo(
                            model_id=m_id,
                            name=name,
                            capability_chat=True,
                            capability_vision=True,  # Claude 3+ has multimodal vision
                            capability_tools=True,
                            capability_json=True,
                            capability_embeddings=False,
                            description=f"Anthropic {name}",
                        )
                    )
                return models
            else:
                raise self._normalize_error(resp.status_code, resp.text)
        except (requests.exceptions.RequestException, AIProviderError):
            # Fallback catalog if models endpoint is restricted on certain tiers
            _logger.info("Using Anthropic standard catalog fallback")
            return [
                ModelInfo("claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet (Hybrid Reasoning)", True, True, True, True),
                ModelInfo("claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet", True, True, True, True),
                ModelInfo("claude-3-5-haiku-20241022", "Claude 3.5 Haiku", True, True, True, True),
                ModelInfo("claude-3-opus-20240229", "Claude 3 Opus", True, True, True, True),
            ]

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
        endpoint = f"{self.base_url}/messages"
        headers = self._get_headers()

        # Separate system messages from user/assistant messages
        system_prompts: List[str] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_prompts.append(content)
            elif role == "assistant" and msg.get("tool_calls"):
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for tool_call in msg["tool_calls"]:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": function.get("name") or tool_call.get("name"),
                        "input": arguments,
                    })
                anthropic_messages.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": str(content or ""),
                    }],
                })
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})

        # Anthropic requires max_tokens
        limit_tokens = max_tokens or 4096

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": anthropic_messages,
            "max_tokens": limit_tokens,
        }

        if system_prompts:
            payload["system"] = "\n\n".join(system_prompts)
        if temperature is not None:
            # Anthropic temperature range is 0.0 to 1.0
            payload["temperature"] = min(max(temperature, 0.0), 1.0)
        if top_p is not None:
            payload["top_p"] = top_p
        if tools:
            # Transform tools if necessary to Anthropic tool schema
            anthropic_tools = []
            for t in tools:
                if t.get("type") == "function" and "function" in t:
                    fn = t["function"]
                    anthropic_tools.append({
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    })
                else:
                    anthropic_tools.append(t)
            payload["tools"] = anthropic_tools

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
            raise AITimeoutError(f"Anthropic request timed out after {self.timeout}s: {te}")
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Network error calling Anthropic: {re}")

        # Parse response blocks
        content_text = ""
        tool_calls = []
        for block in res_json.get("content", []):
            b_type = block.get("type")
            if b_type == "text":
                content_text += block.get("text", "")
            elif b_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return AIResponse(
            content=content_text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            finish_reason=res_json.get("stop_reason"),
            raw=res_json,
        )
