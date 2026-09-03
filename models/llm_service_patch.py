# -*- coding: utf-8 -*-
"""Bridge custom provider records into Odoo 19's native ``LLMApiService``.

Odoo's AI service is a plain Python class rather than an ORM model, so it does
not provide a registry inheritance hook.  Keep the patch deliberately small:
provider/model resolution and the two network operations (chat + embeddings).
The native agent loop, tool execution, RAG, logging context and UI remain Odoo's.
"""

import logging

from odoo.api import Environment
from odoo.exceptions import UserError

from ..services import (
    AIProviderError,
    build_messages,
    custom_record_id,
    format_tools,
    parse_tool_calls,
    response_format_for_schema,
    tool_result_message,
)

_logger = logging.getLogger(__name__)

try:
    import odoo.addons.ai.models.ai_embedding as ai_embedding_model
    import odoo.addons.ai.utils.llm_api_service as llm_api_service
    import odoo.addons.ai.utils.llm_providers as llm_providers

    AI_MODULE_AVAILABLE = True
except ImportError:
    AI_MODULE_AVAILABLE = False
    _logger.warning("Odoo native AI modules are unavailable; custom provider bridge was not installed.")


def _is_enabled(env):
    value = env['ir.config_parameter'].sudo().get_param(
        'odoo_custom_llm_provider.use_custom_ai_provider', 'False'
    )
    return value in ('True', 'true', '1', True)


def _active_provider(env):
    if not _is_enabled(env):
        return env['ai.provider']
    provider_id = env['ir.config_parameter'].sudo().get_param(
        'odoo_custom_llm_provider.active_provider_id'
    )
    try:
        provider = env['ai.provider'].sudo().browse(int(provider_id)).exists()
    except (TypeError, ValueError):
        return env['ai.provider']
    return provider if provider and provider.active else env['ai.provider']


def _model_from_value(env, value):
    model_id = custom_record_id(value)
    if model_id is None:
        return env['ai.model']
    model = env['ai.model'].sudo().browse(model_id).exists()
    if model and model.active and model.provider_id.active and model.capability_chat:
        return model
    return env['ai.model']


def _request_argument(args, kwargs, name, index, default=None):
    return kwargs[name] if name in kwargs else (args[index] if len(args) > index else default)


def _patch_provider_helpers():
    original_get_provider = llm_providers.get_provider
    original_get_embedding_provider = llm_providers.get_provider_for_embedding_model
    original_get_embedding_config = llm_providers.get_embedding_config
    original_check_deprecation = llm_providers.check_model_depreciation

    def get_provider(env, llm_model):
        model = _model_from_value(env, llm_model)
        if model:
            return f"custom_{model.provider_id.id}"
        return original_get_provider(env, llm_model)

    def get_provider_for_embedding_model(env, embedding_model):
        active = _active_provider(env)
        if active and active.embedding_model == embedding_model:
            return f"custom_{active.id}"
        provider = env['ai.provider'].sudo().search([
            ('active', '=', True),
        ]).filtered(lambda item: item.embedding_model == embedding_model)[:1]
        if provider:
            return f"custom_{provider.id}"
        return original_get_embedding_provider(env, embedding_model)

    def get_embedding_config(env, provider):
        if custom_record_id(provider) is not None:
            return {'max_batch_size': 100, 'max_tokens_per_request': 100000}
        return original_get_embedding_config(env, provider)

    def check_model_depreciation(env, model):
        if custom_record_id(model) is not None:
            return None
        return original_check_deprecation(env, model)

    llm_providers.get_provider = get_provider
    llm_providers.get_provider_for_embedding_model = get_provider_for_embedding_model
    llm_providers.get_embedding_config = get_embedding_config
    llm_providers.check_model_depreciation = check_model_depreciation

    # ai.models.ai_embedding imports these helpers directly, so patch its local
    # references too. Merely replacing llm_providers.* does not affect them.
    ai_embedding_model.get_provider_for_embedding_model = get_provider_for_embedding_model
    ai_embedding_model.get_embedding_config = get_embedding_config


def _patch_llm_api_service():
    service_class = llm_api_service.LLMApiService
    original_init = service_class.__init__
    original_get_api_token = service_class._get_api_token
    original_request_llm = service_class._request_llm
    original_request_llm_silent = service_class._request_llm_silent
    original_build_tool_result = service_class._build_tool_call_response
    original_get_embedding = service_class.get_embedding

    def patched_init(self, env: Environment, provider: str = 'openai') -> None:
        provider_record_id = custom_record_id(provider)
        self.custom_provider = None
        self.custom_override_native = False

        if provider_record_id is not None:
            self.env = env
            self.provider = provider
            custom_provider = env['ai.provider'].sudo().browse(provider_record_id).exists()
            if not custom_provider or not custom_provider.active:
                raise UserError(env._("The selected custom AI provider is missing or inactive."))
            self.custom_provider = custom_provider
            self.base_url = custom_provider.base_url
            return

        original_init(self, env, provider)
        custom_provider = _active_provider(env)
        if custom_provider:
            self.custom_provider = custom_provider
            self.custom_override_native = True

    def patched_get_api_token(self):
        if self.custom_provider and not self.custom_override_native:
            return self.custom_provider.api_key or ""
        return original_get_api_token(self)

    def _resolve_request_model(self, requested_model):
        explicit_model = _model_from_value(self.env, requested_model)
        if explicit_model:
            if explicit_model.provider_id != self.custom_provider:
                raise UserError(self.env._("The selected model does not belong to the resolved custom provider."))
            return explicit_model

        if self.custom_override_native:
            model = self.custom_provider.default_model_id
            if not model or not model.active or not model.capability_chat:
                raise UserError(self.env._(
                    "The active custom AI provider has no valid default chat model. "
                    "Select one in AI Settings."
                ))
            return model

        # Compatibility for callers that already resolved a custom provider but
        # pass the provider's technical model name instead of custom_<record id>.
        model = self.env['ai.model'].sudo().search([
            ('provider_id', '=', self.custom_provider.id),
            ('model_id', '=', requested_model),
            ('active', '=', True),
            ('capability_chat', '=', True),
        ], limit=1)
        if not model:
            raise UserError(self.env._("No active custom AI model matches '%s'.", requested_model))
        return model

    def _log_request(self, model, response=None, messages=None, error=None):
        try:
            prompt = next(
                (str(item.get('content') or '')[:300] for item in reversed(messages or ())
                 if item.get('role') == 'user'),
                '',
            )
            values = {
                'user_id': self.env.user.id,
                'provider_id': self.custom_provider.id,
                'model_id': model.id if model else False,
                'use_case': 'discuss_assistant',
                'status': 'error' if error else 'success',
                'prompt_snippet': prompt,
                'error_message': str(error)[:1000] if error else False,
            }
            if response:
                values.update({
                    'prompt_tokens': response.prompt_tokens,
                    'completion_tokens': response.completion_tokens,
                    'total_tokens': response.total_tokens,
                    'latency_ms': response.latency_ms,
                    'cost_estimated': model.compute_cost(
                        response.prompt_tokens, response.completion_tokens
                    ) if model else 0.0,
                    'response_snippet': (response.content or '')[:300],
                })
            self.env['ai.request.log'].sudo().create(values)
        except Exception as log_error:
            _logger.warning("Could not create custom AI request log: %s", log_error)

    def patched_request_llm(self, *args, **kwargs):
        if not self.custom_provider:
            return original_request_llm(self, *args, **kwargs)

        requested_model = _request_argument(args, kwargs, 'llm_model', 0)
        system_prompts = _request_argument(args, kwargs, 'system_prompts', 1, [])
        user_prompts = _request_argument(args, kwargs, 'user_prompts', 2, [])
        tools = _request_argument(args, kwargs, 'tools', 3)
        files = _request_argument(args, kwargs, 'files', 4)
        schema = _request_argument(args, kwargs, 'schema', 5)
        temperature = _request_argument(args, kwargs, 'temperature', 6, 0.2)
        inputs = _request_argument(args, kwargs, 'inputs', 7, ())
        web_grounding = _request_argument(args, kwargs, 'web_grounding', 8, False)

        model = _resolve_request_model(self, requested_model)
        messages = build_messages(system_prompts, user_prompts, inputs, files)
        if web_grounding:
            _logger.info(
                "Web grounding was requested but is provider-specific; continuing with custom provider %s",
                self.custom_provider.name,
            )

        try:
            response = self.custom_provider.get_adapter().chat_completion(
                model_id=model.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=model.max_tokens,
                top_p=model.top_p,
                tools=format_tools(tools),
                response_format=response_format_for_schema(schema),
            )
        except AIProviderError as error:
            _log_request(self, model, messages=messages, error=error)
            raise UserError(str(error)) from error

        next_actions = parse_tool_calls(response.tool_calls)
        next_inputs = list(inputs or ())
        if next_actions:
            next_inputs.append({
                'role': 'assistant',
                'content': response.content or None,
                'tool_calls': response.tool_calls,
            })
        _log_request(self, model, response=response, messages=messages)
        return ([response.content] if response.content else []), next_actions, next_inputs

    def patched_request_llm_silent(self, *args, **kwargs):
        # Odoo pre-converts every input to Gemini whenever ``provider`` equals
        # ``google``. With global replacement that conversion would corrupt the
        # payload before an OpenAI-compatible/Anthropic/Ollama adapter sees it.
        if self.custom_provider and self.custom_override_native and self.provider == 'google':
            native_provider = self.provider
            self.provider = f'custom_{self.custom_provider.id}'
            try:
                return original_request_llm_silent(self, *args, **kwargs)
            finally:
                self.provider = native_provider
        return original_request_llm_silent(self, *args, **kwargs)

    def patched_build_tool_call_response(self, tool_call_id, return_value):
        if self.custom_provider:
            return tool_result_message(
                self.custom_provider.provider_type, tool_call_id, return_value
            )
        return original_build_tool_result(self, tool_call_id, return_value)

    def patched_get_embedding(
        self, input, dimensions, model='text-embedding-3-small',
        encoding_format=None, user=None,
    ):
        if not self.custom_provider:
            return original_get_embedding(
                self, input, dimensions, model=model,
                encoding_format=encoding_format, user=user,
            )
        embedding_model = self.custom_provider.embedding_model
        if not embedding_model:
            raise UserError(self.env._(
                "Configure an embedding model for custom provider '%s'.",
                self.custom_provider.name,
            ))
        try:
            return self.custom_provider.get_adapter().get_embedding(
                input=input,
                dimensions=dimensions,
                model=embedding_model,
                encoding_format=encoding_format,
                user=user,
            )
        except AIProviderError as error:
            raise UserError(str(error)) from error

    service_class.__init__ = patched_init
    service_class._get_api_token = patched_get_api_token
    service_class._request_llm = patched_request_llm
    service_class._request_llm_silent = patched_request_llm_silent
    service_class._build_tool_call_response = patched_build_tool_call_response
    service_class.get_embedding = patched_get_embedding
    service_class._custom_provider_patched = True


def _patch_ai_system():
    if not AI_MODULE_AVAILABLE:
        return
    service_class = llm_api_service.LLMApiService
    if getattr(service_class, '_custom_provider_patched', False):
        return
    _patch_provider_helpers()
    _patch_llm_api_service()
    _logger.info("Installed Odoo native AI bridge for custom LLM providers.")


_patch_ai_system()
