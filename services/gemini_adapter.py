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


class GeminiAdapter(BaseAIAdapter):
    """
    Adapter for Google Gemini REST API (https://generativelanguage.googleapis.com/v1beta).
    Supports dynamic model discovery, native Gemini contents schema, and systemInstructions.
    """

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
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
                models = data.get("models", [])
                return True, f"Google Gemini connection successful! Discovered {len(models)} models."
            elif resp.status_code in (400, 401, 403):
                return False, f"Authentication failed: Invalid Google Gemini API Key ({resp.status_code})"
            else:
                return False, f"Gemini error {resp.status_code}: {resp.text[:300]}"
        except requests.exceptions.Timeout:
            return False, f"Connection timed out connecting to {self.base_url}"
        except Exception as e:
            return False, f"Gemini connection error: {str(e)}"

    def fetch_available_models(self) -> List[ModelInfo]:
        endpoint = f"{self.base_url}/models"
        try:
            resp = requests.get(endpoint, headers=self._get_headers(), timeout=self.timeout)
            if resp.status_code != 200:
                raise self._normalize_error(resp.status_code, resp.text)
            data = resp.json()
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Network error fetching Gemini models: {re}")

        models = []
        for item in data.get("models", []):
            methods = item.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue

            raw_name = item.get("name", "")  # e.g., 'models/gemini-2.5-flash'
            clean_id = raw_name.replace("models/", "")
            disp_name = item.get("displayName") or clean_id
            desc = item.get("description", "")

            models.append(
                ModelInfo(
                    model_id=clean_id,
                    name=disp_name,
                    capability_chat=True,
                    capability_vision=True,  # Gemini 1.5/2.0+ are natively multimodal
                    capability_tools=True,
                    capability_json=True,
                    capability_embeddings="embedContent" in methods,
                    description=desc,
                )
            )
        return models

    def get_embedding(
        self, input, dimensions: Optional[int], model: str,
        encoding_format: Optional[str] = None, user: Optional[str] = None,
    ) -> Dict[str, Any]:
        values = [input] if isinstance(input, str) else input
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise AIProviderError("Gemini embeddings require text input")

        clean_model = model.replace("models/", "")
        model_path = f"models/{clean_model}"
        requests_payload = []
        for value in values:
            item = {
                "model": model_path,
                "content": {"parts": [{"text": value}]},
            }
            if dimensions:
                item["outputDimensionality"] = dimensions
            requests_payload.append(item)

        endpoint = f"{self.base_url}/models/{clean_model}:batchEmbedContents"
        try:
            response = requests.post(
                endpoint,
                headers=self._get_headers(),
                json={"requests": requests_payload},
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise self._normalize_error(response.status_code, response.text)
            payload = response.json()
        except requests.exceptions.Timeout as error:
            raise AITimeoutError(f"Gemini embedding request timed out: {error}")
        except requests.exceptions.RequestException as error:
            raise AIProviderError(f"Gemini embedding request failed: {error}")

        embeddings = payload.get("embeddings") or []
        result = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": item.get("values", []),
                }
                for index, item in enumerate(embeddings)
            ],
            "model": clean_model,
            "usage": {},
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
        clean_model_id = model_id.replace("models/", "")
        endpoint = f"{self.base_url}/models/{clean_model_id}:generateContent"
        headers = self._get_headers()

        # Transform messages into Gemini format
        system_texts: List[str] = []
        gemini_contents: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content") or ""
            if role == "system":
                system_texts.append(str(content))
            elif msg.get("parts"):
                gemini_contents.append({
                    "role": "model" if role in ("assistant", "model") else "user",
                    "parts": msg["parts"],
                })
            elif role == "assistant" and msg.get("tool_calls"):
                parts = []
                if content:
                    parts.append({"text": str(content)})
                for tool_call in msg["tool_calls"]:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    parts.append({
                        "functionCall": {
                            "name": function.get("name") or tool_call.get("name"),
                            "args": arguments,
                        },
                    })
                gemini_contents.append({"role": "model", "parts": parts})
            elif role == "user":
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:") and ";base64," in image_url:
                                metadata, data = image_url.split(",", 1)
                                parts.append({
                                    "inline_data": {
                                        "mime_type": metadata[5:].split(";", 1)[0],
                                        "data": data,
                                    },
                                })
                else:
                    parts = [{"text": str(content)}]
                gemini_contents.append({
                    "role": "user",
                    "parts": parts,
                })
            elif role in ("assistant", "model"):
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": str(content)}],
                })
            elif role == "tool":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": msg.get("tool_call_id"),
                            "response": {"result": str(content)},
                        },
                    }],
                })

        payload: Dict[str, Any] = {
            "contents": gemini_contents,
        }

        if system_texts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_texts)}]
            }

        generation_config: Dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if top_p is not None:
            generation_config["topP"] = top_p
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            generation_config["responseMimeType"] = "application/json"
            if response_format.get("type") == "json_schema":
                generation_config["responseJsonSchema"] = response_format["json_schema"]["schema"]

        if generation_config:
            payload["generationConfig"] = generation_config

        if tools:
            payload["tools"] = {
                "functionDeclarations": [
                    {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": tool["function"].get("parameters", {}),
                    }
                    for tool in tools
                    if tool.get("type") == "function" and tool.get("function")
                ],
            }

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
            raise AITimeoutError(f"Gemini request timed out after {self.timeout}s: {te}")
        except requests.exceptions.RequestException as re:
            raise AIProviderError(f"Network error calling Gemini: {re}")

        candidates = res_json.get("candidates", [])
        if not candidates:
            raise AIProviderError("Gemini returned empty candidates list", details=res_json)

        first_cand = candidates[0]
        content_obj = first_cand.get("content", {})
        parts = content_obj.get("parts", [])
        output_text = "".join(p.get("text", "") for p in parts)
        tool_calls = []
        for part in parts:
            if function_call := part.get("functionCall"):
                tool_calls.append({
                    "id": function_call.get("name"),
                    "type": "function",
                    "function": {
                        "name": function_call.get("name"),
                        "arguments": json.dumps(function_call.get("args") or {}),
                    },
                })
        finish_reason = first_cand.get("finishReason")

        usage = res_json.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

        return AIResponse(
            content=output_text,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            raw=res_json,
        )
