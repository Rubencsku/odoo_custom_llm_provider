# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import patch, MagicMock

import requests

import os
import sys

_services_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _services_path not in sys.path:
    sys.path.insert(0, _services_path)

from services.base_adapter import (
    AIResponse,
    AIProviderError,
    AIAuthenticationError,
    AIRateLimitError,
    AITimeoutError,
    AIServiceUnavailableError,
)
from services.openai_adapter import OpenAICompatibleAdapter
from services.anthropic_adapter import AnthropicAdapter
from services.gemini_adapter import GeminiAdapter
from services.ollama_adapter import OllamaAdapter
from services import get_adapter




class TestAIAdapters(unittest.TestCase):

    # ---------------------------------------------------------
    # Factory Tests
    # ---------------------------------------------------------
    def test_adapter_factory(self):
        adapter_openai = get_adapter('openai', 'https://api.openai.com/v1', 'sk-test')
        self.assertIsInstance(adapter_openai, OpenAICompatibleAdapter)

        adapter_anthropic = get_adapter('anthropic', 'https://api.anthropic.com/v1', 'sk-ant')
        self.assertIsInstance(adapter_anthropic, AnthropicAdapter)

        adapter_gemini = get_adapter('gemini', 'https://generativelanguage.googleapis.com/v1beta', 'key-gem')
        self.assertIsInstance(adapter_gemini, GeminiAdapter)

        adapter_ollama = get_adapter('ollama', 'http://localhost:11434')
        self.assertIsInstance(adapter_ollama, OllamaAdapter)

    # ---------------------------------------------------------
    # OpenAI Compatible Adapter Tests
    # ---------------------------------------------------------
    @patch('requests.get')
    def test_openai_test_connection_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}
        mock_get.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://api.openai.com/v1", api_key="sk-test")
        success, msg = adapter.test_connection()
        self.assertTrue(success)
        self.assertIn("Retrieved 2 models", msg)

    @patch('requests.get')
    def test_openai_fetch_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "gpt-4o", "name": "GPT-4o Omnimodel"},
                {"id": "text-embedding-3-small", "name": "Embeddings"},
            ]
        }
        mock_get.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://api.openai.com/v1", api_key="sk-test")
        models = adapter.fetch_available_models()
        self.assertEqual(len(models), 2)
        gpt4o = next(m for m in models if m.model_id == "gpt-4o")
        self.assertTrue(gpt4o.capability_chat)
        self.assertTrue(gpt4o.capability_vision)

        embed = next(m for m in models if m.model_id == "text-embedding-3-small")
        self.assertTrue(embed.capability_embeddings)
        self.assertFalse(embed.capability_chat)

    @patch('requests.post')
    def test_openai_chat_completion_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello from OpenAI!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
        mock_post.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://api.openai.com/v1", api_key="sk-test")
        res = adapter.chat_completion(
            model_id="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7,
        )

        self.assertEqual(res.content, "Hello from OpenAI!")
        self.assertEqual(res.prompt_tokens, 12)
        self.assertEqual(res.completion_tokens, 8)
        self.assertEqual(res.total_tokens, 20)

    @patch('requests.post')
    def test_openai_embeddings_use_configured_endpoint(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "custom-embed",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        mock_post.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://llm.example/v1", api_key="secret")
        result = adapter.get_embedding("hello", 2, "custom-embed")

        self.assertEqual(result["data"][0]["embedding"], [0.1, 0.2])
        self.assertEqual(mock_post.call_args.args[0], "https://llm.example/v1/embeddings")
        self.assertEqual(mock_post.call_args.kwargs["json"]["model"], "custom-embed")

    @patch('requests.post')
    def test_openai_rate_limit_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = '{"error": {"message": "Rate limit reached"}}'
        mock_post.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://api.openai.com/v1", api_key="sk-test")
        with self.assertRaises(AIRateLimitError):
            adapter.chat_completion(
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

    # ---------------------------------------------------------
    # Anthropic Adapter Tests
    # ---------------------------------------------------------
    @patch('requests.post')
    def test_anthropic_chat_completion(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "Greetings from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 7,
            },
        }
        mock_post.return_value = mock_resp

        adapter = AnthropicAdapter("https://api.anthropic.com/v1", api_key="sk-ant-test")
        res = adapter.chat_completion(
            model_id="claude-3-7-sonnet-20250219",
            messages=[
                {"role": "system", "content": "Act like an expert."},
                {"role": "user", "content": "Hello"},
            ],
            max_tokens=1000,
        )

        self.assertEqual(res.content, "Greetings from Claude!")
        self.assertEqual(res.prompt_tokens, 15)
        self.assertEqual(res.completion_tokens, 7)
        self.assertEqual(res.total_tokens, 22)

        # Validate that system message was moved to root of payload
        called_payload = mock_post.call_args[1]["json"]
        self.assertEqual(called_payload["system"], "Act like an expert.")
        self.assertEqual(len(called_payload["messages"]), 1)
        self.assertEqual(called_payload["messages"][0]["role"], "user")

    # ---------------------------------------------------------
    # Gemini Adapter Tests
    # ---------------------------------------------------------
    @patch('requests.post')
    def test_gemini_chat_completion(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello from Google Gemini!"}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 6,
                "totalTokenCount": 16,
            },
        }
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter("https://generativelanguage.googleapis.com/v1beta", api_key="gem-key")
        res = adapter.chat_completion(
            model_id="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are Gemini."},
                {"role": "user", "content": "Hi!"},
            ],
        )

        self.assertEqual(res.content, "Hello from Google Gemini!")
        self.assertEqual(res.prompt_tokens, 10)
        self.assertEqual(res.completion_tokens, 6)
        self.assertEqual(res.total_tokens, 16)

        # Check formatting of payload
        called_payload = mock_post.call_args[1]["json"]
        self.assertIn("systemInstruction", called_payload)
        self.assertEqual(called_payload["contents"][0]["role"], "user")

    @patch('requests.post')
    def test_gemini_tool_call_is_returned_to_odoo(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "lookup_partner", "args": {"id": 7}},
                }]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {},
        }
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter("https://generativelanguage.googleapis.com/v1beta", api_key="key")
        result = adapter.chat_completion(
            model_id="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Find partner 7"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "lookup_partner",
                    "description": "Find a partner",
                    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
                },
            }],
        )

        self.assertEqual(result.tool_calls[0]["function"]["name"], "lookup_partner")
        self.assertIn("functionDeclarations", mock_post.call_args.kwargs["json"]["tools"])

    @patch('requests.post')
    def test_gemini_embeddings_are_normalized_for_odoo(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}],
        }
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter("https://generativelanguage.googleapis.com/v1beta", api_key="key")
        result = adapter.get_embedding(["one", "two"], 2, "gemini-embedding-001")

        self.assertEqual(len(result["data"]), 2)
        self.assertEqual(result["data"][1]["embedding"], [0.3, 0.4])
        self.assertIn(":batchEmbedContents", mock_post.call_args.args[0])

    # ---------------------------------------------------------
    # Ollama Adapter Tests
    # ---------------------------------------------------------
    @patch('requests.get')
    def test_ollama_fetch_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:32b", "model": "qwen2.5:32b"},
                {"name": "llava:latest", "model": "llava:latest"},
            ]
        }
        mock_get.return_value = mock_resp

        adapter = OllamaAdapter("http://localhost:11434")
        models = adapter.fetch_available_models()
        self.assertEqual(len(models), 2)
        llava = next(m for m in models if "llava" in m.model_id)
        self.assertTrue(llava.capability_vision)

    @patch('requests.post')
    def test_ollama_chat_completion(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"role": "assistant", "content": "Local inference response"},
            "prompt_eval_count": 25,
            "eval_count": 15,
            "total_duration": 450000000,  # 450ms
        }
        mock_post.return_value = mock_resp

        adapter = OllamaAdapter("http://localhost:11434")
        res = adapter.chat_completion(
            model_id="qwen2.5:32b",
            messages=[{"role": "user", "content": "Local test"}],
        )

        self.assertEqual(res.content, "Local inference response")
        self.assertEqual(res.prompt_tokens, 25)
        self.assertEqual(res.completion_tokens, 15)
        self.assertEqual(res.latency_ms, 450.0)

    @patch('requests.post')
    def test_ollama_embeddings_are_normalized_for_odoo(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "prompt_eval_count": 4,
        }
        mock_post.return_value = mock_resp

        adapter = OllamaAdapter("http://localhost:11434")
        result = adapter.get_embedding(["one", "two"], 3, "nomic-embed-text")

        self.assertEqual(result["data"][0]["embedding"], [0.1, 0.2, 0.0])
        self.assertEqual(mock_post.call_args.args[0], "http://localhost:11434/api/embed")

    # ---------------------------------------------------------
    # Resiliency, Errors & Custom Headers Tests
    # ---------------------------------------------------------
    @patch('requests.post')
    def test_service_unavailable_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.text = 'Service Temporarily Overloaded'
        mock_post.return_value = mock_resp

        adapter = OpenAICompatibleAdapter("https://api.openai.com/v1", api_key="sk-test")
        with self.assertRaises(AIServiceUnavailableError):
            adapter.chat_completion(
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

    @patch('requests.post')
    def test_custom_headers_forwarded(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        mock_post.return_value = mock_resp

        adapter = OpenAICompatibleAdapter(
            "https://openrouter.ai/api/v1",
            api_key="sk-or",
            custom_headers={"HTTP-Referer": "https://company.odoo.com", "X-Title": "Odoo ERP"},
        )
        adapter.chat_completion(
            model_id="anthropic/claude-3.5-sonnet",
            messages=[{"role": "user", "content": "Hi"}],
        )

        headers_used = mock_post.call_args[1]["headers"]
        self.assertEqual(headers_used.get("HTTP-Referer"), "https://company.odoo.com")
        self.assertEqual(headers_used.get("X-Title"), "Odoo ERP")
        self.assertEqual(headers_used.get("Authorization"), "Bearer sk-or")

    @patch('requests.post')
    def test_gemini_json_mode_configuration(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "{\"result\": true}"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5, "totalTokenCount": 10},
        }
        mock_post.return_value = mock_resp

        adapter = GeminiAdapter("https://generativelanguage.googleapis.com/v1beta", api_key="key")
        adapter.chat_completion(
            model_id="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Give me JSON"}],
            response_format={"type": "json_object"},
        )

        called_json = mock_post.call_args[1]["json"]
        self.assertEqual(called_json["generationConfig"]["responseMimeType"], "application/json")


if __name__ == '__main__':
    unittest.main()
