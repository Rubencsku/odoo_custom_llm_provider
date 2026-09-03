# -*- coding: utf-8 -*-
import logging
from odoo import models, api, _
from odoo.exceptions import UserError

from ..services import (
    AIResponse,
    AIProviderError,
    AIRateLimitError,
    AITimeoutError,
    AIServiceUnavailableError,
)

_logger = logging.getLogger(__name__)


class AIService(models.AbstractModel):
    _name = 'ai.service'
    _description = 'Central AI Service Router & Fallback Engine'

    @api.model
    def is_custom_ai_enabled(self) -> bool:
        """Check if custom provider routing is enabled in settings."""
        Param = self.env['ir.config_parameter'].sudo()
        val = Param.get_param('odoo_custom_llm_provider.use_custom_ai_provider', 'False')
        return val in ('True', 'true', '1', True)

    @api.model
    def get_model_for_use_case(self, use_case: str):
        """Retrieve the configured ai.model record for a given functional use case."""
        Param = self.env['ir.config_parameter'].sudo()
        key_map = {
            'discuss_assistant': 'odoo_custom_llm_provider.discuss_assistant_model_id',
            'livechat': 'odoo_custom_llm_provider.livechat_model_id',
            'chatter_summary': 'odoo_custom_llm_provider.chatter_summary_model_id',
            'web_editor': 'odoo_custom_llm_provider.web_editor_model_id',
            'document_ocr': 'odoo_custom_llm_provider.document_ocr_model_id',
        }

        param_key = key_map.get(use_case)
        if param_key:
            model_id_val = Param.get_param(param_key)
            if model_id_val:
                try:
                    m = self.env['ai.model'].sudo().browse(int(model_id_val))
                    if m.exists() and m.active and m.provider_id.active:
                        return m
                except (ValueError, TypeError):
                    pass

        # Fallback to any active chat model from active provider or any active provider
        active_provider_id = Param.get_param('odoo_custom_llm_provider.active_provider_id')
        if active_provider_id:
            try:
                m = self.env['ai.model'].sudo().search([
                    ('provider_id', '=', int(active_provider_id)),
                    ('capability_chat', '=', True),
                    ('active', '=', True),
                ], limit=1)
                if m:
                    return m
            except (ValueError, TypeError):
                pass

        # Global fallback: first active chat model
        return self.env['ai.model'].sudo().search([
            ('capability_chat', '=', True),
            ('active', '=', True),
            ('provider_id.active', '=', True),
        ], limit=1)

    @api.model
    def get_fallback_model(self, exclude_model_id: int = None):
        """Retrieve the configured secondary fallback model."""
        Param = self.env['ir.config_parameter'].sudo()
        fallback_val = Param.get_param('odoo_custom_llm_provider.fallback_model_id')
        if fallback_val:
            try:
                fm = self.env['ai.model'].sudo().browse(int(fallback_val))
                if fm.exists() and fm.active and fm.provider_id.active and fm.id != exclude_model_id:
                    return fm
            except (ValueError, TypeError):
                pass
        return self.env['ai.model']

    @api.model
    def execute_chat(
        self,
        use_case: str,
        messages: list,
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        tools: list = None,
        response_format: dict = None,
        **kwargs
    ) -> AIResponse:
        """
        Execute chat completion with automatic fallback cascade and audit logging.
        """
        model = self.get_model_for_use_case(use_case)
        if not model:
            raise UserError(_("No active AI model is configured for use case '%s'. Please configure one in General Settings.") % use_case)

        fallback_model = self.get_fallback_model(exclude_model_id=model.id)

        # Prepare snippet for logging
        prompt_snippet = ""
        for m in reversed(messages):
            if m.get('role') in ('user', 'system'):
                content = str(m.get('content', ''))
                prompt_snippet = content[:300]
                break

        used_fallback = False
        target_model = model
        error_msg = ""

        try:
            # Primary attempt
            response = target_model.generate_chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                tools=tools,
                response_format=response_format,
                **kwargs
            )
        except (AIRateLimitError, AITimeoutError, AIServiceUnavailableError) as recoverable_err:
            _logger.warning(
                "Primary AI Model '%s' failed with recoverable error: %s. Attempting fallback.",
                model.name, recoverable_err
            )
            if fallback_model:
                try:
                    used_fallback = True
                    target_model = fallback_model
                    response = target_model.generate_chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        tools=tools,
                        response_format=response_format,
                        **kwargs
                    )
                except Exception as fallback_err:
                    error_msg = f"Primary failed ({recoverable_err}). Fallback ({fallback_model.name}) also failed: {fallback_err}"
                    self._log_transaction(
                        user_id=self.env.user.id,
                        provider_id=fallback_model.provider_id.id,
                        model_id=fallback_model.id,
                        use_case=use_case,
                        status='error',
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        latency_ms=0.0,
                        cost=0.0,
                        prompt_snippet=prompt_snippet,
                        response_snippet="",
                        error_message=error_msg,
                    )
                    raise UserError(_("All AI providers failed. %s") % error_msg)
            else:
                error_msg = str(recoverable_err)
                self._log_transaction(
                    user_id=self.env.user.id,
                    provider_id=model.provider_id.id,
                    model_id=model.id,
                    use_case=use_case,
                    status='error',
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    latency_ms=0.0,
                    cost=0.0,
                    prompt_snippet=prompt_snippet,
                    response_snippet="",
                    error_message=error_msg,
                )
                raise UserError(_("AI Provider Error: %s") % error_msg)
        except Exception as general_err:
            error_msg = str(general_err)
            self._log_transaction(
                user_id=self.env.user.id,
                provider_id=model.provider_id.id,
                model_id=model.id,
                use_case=use_case,
                status='error',
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=0.0,
                cost=0.0,
                prompt_snippet=prompt_snippet,
                response_snippet="",
                error_message=error_msg,
            )
            raise UserError(_("AI Provider execution failed: %s") % error_msg)

        # Log successful call
        cost = target_model.compute_cost(response.prompt_tokens, response.completion_tokens)
        status = 'fallback_used' if used_fallback else 'success'

        self._log_transaction(
            user_id=self.env.user.id,
            provider_id=target_model.provider_id.id,
            model_id=target_model.id,
            use_case=use_case,
            status=status,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_ms=response.latency_ms,
            cost=cost,
            prompt_snippet=prompt_snippet,
            response_snippet=(response.content or "")[:300],
            error_message="",
        )

        return response

    @api.model
    def _log_transaction(
        self,
        user_id,
        provider_id,
        model_id,
        use_case,
        status,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        latency_ms,
        cost,
        prompt_snippet,
        response_snippet,
        error_message,
    ):
        """Safely record token and request statistics into ai.request.log."""
        try:
            self.env['ai.request.log'].sudo().create({
                'user_id': user_id,
                'provider_id': provider_id,
                'model_id': model_id,
                'use_case': use_case,
                'status': status,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'latency_ms': latency_ms,
                'cost_estimated': cost,
                'prompt_snippet': prompt_snippet,
                'response_snippet': response_snippet,
                'error_message': error_message,
            })
        except Exception as e:
            _logger.error("Failed to write to ai.request.log: %s", e)


class AIServiceMixin(models.AbstractModel):
    _name = 'ai.service.mixin'
    _description = 'AI Service Mixin for Odoo Models'

    def ai_complete_prompt(self, prompt: str, system_prompt: str = None, use_case: str = 'custom', **kwargs) -> str:
        """
        Helper method allowing any record/model to get an AI completion effortlessly:
        reply = record.ai_complete_prompt("Summarize this ticket...", use_case="chatter_summary")
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.env['ai.service'].execute_chat(
            use_case=use_case,
            messages=messages,
            **kwargs
        )
        return response.content
