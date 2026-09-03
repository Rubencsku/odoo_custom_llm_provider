# -*- coding: utf-8 -*-
import logging
from odoo import models, fields, api, _
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)


class AIAgent(models.Model):
    _inherit = 'ai.agent'

    @api.model
    def _get_llm_model_selection(self):
        """
        Extend native Odoo AI agent model selection to include all custom models
        discovered from active AI Providers.
        """
        selection = super()._get_llm_model_selection()
        existing_keys = {item[0] for item in selection}

        try:
            custom_models = self.env['ai.model'].sudo().search([
                ('active', '=', True),
                ('capability_chat', '=', True),
                ('provider_id.active', '=', True),
            ])
            for m in custom_models:
                key = f"custom_{m.id}"
                label = f"{m.name} [{m.provider_id.name}]"
                if key not in existing_keys:
                    selection.append((key, label))
                    existing_keys.add(key)
        except Exception as e:
            _logger.warning("Could not load custom AI models for selection: %s", e)

        return selection

    def _resolve_custom_model(self):
        """Helper to find the ai.model record corresponding to self.llm_model."""
        self.ensure_one()
        model_val = self.llm_model
        if not model_val:
            return self.env['ai.model']

        # 1. custom_{id} format
        if str(model_val).startswith('custom_'):
            try:
                m_id = int(str(model_val).split('_')[1])
                m = self.env['ai.model'].sudo().browse(m_id)
                if m.exists():
                    return m
            except (ValueError, IndexError):
                pass

        return self.env['ai.model']

    def _get_provider(self):
        """
        Return the provider name or custom provider identifier for the agent's model.
        """
        self.ensure_one()
        custom_m = self._resolve_custom_model()
        if custom_m:
            return f"custom_{custom_m.provider_id.id}"
        return super()._get_provider()

    def _get_embedding_model(self):
        """
        Return the embedding model associated with the custom provider or fallback.
        """
        self.ensure_one()
        custom_m = self._resolve_custom_model()
        if custom_m:
            prov = custom_m.provider_id
            if prov.embedding_model:
                return prov.embedding_model
            return "text-embedding-3-small"

        enabled = self.env['ir.config_parameter'].sudo().get_param(
            'odoo_custom_llm_provider.use_custom_ai_provider', 'False'
        )
        if enabled in ('True', 'true', '1', True):
            provider_id = self.env['ir.config_parameter'].sudo().get_param(
                'odoo_custom_llm_provider.active_provider_id'
            )
            if provider_id:
                provider = self.env['ai.provider'].sudo().browse(int(provider_id)).exists()
                if provider and provider.active and provider.embedding_model:
                    return provider.embedding_model
        return super()._get_embedding_model()
