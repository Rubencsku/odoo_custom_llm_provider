# -*- coding: utf-8 -*-
from odoo import models, fields, api, _
from odoo.exceptions import UserError


class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    use_custom_ai_provider = fields.Boolean(
        string='Use a Custom AI Provider',
        config_parameter='odoo_custom_llm_provider.use_custom_ai_provider',
        default=False,
        help='Route Odoo native AI requests through the selected custom provider.',
    )

    ai_active_provider_id = fields.Many2one(
        comodel_name='ai.provider',
        string='Active AI Provider',
        config_parameter='odoo_custom_llm_provider.active_provider_id',
        help='Primary provider used to configure and route LLM queries.',
    )

    # In-Settings configuration fields mapped to the active provider
    ai_active_provider_type = fields.Selection(
        related='ai_active_provider_id.provider_type',
        string='Provider Type',
        readonly=True,
    )
    ai_active_provider_base_url = fields.Char(
        related='ai_active_provider_id.base_url',
        string='Provider Base URL',
        readonly=False,
    )
    ai_active_provider_api_key = fields.Char(
        related='ai_active_provider_id.api_key',
        string='Provider API Key',
        readonly=False,
    )
    ai_active_provider_status = fields.Selection(
        related='ai_active_provider_id.status',
        string='Provider Status',
        readonly=True,
    )
    ai_active_provider_status_message = fields.Text(
        related='ai_active_provider_id.status_message',
        string='Diagnostic Status',
        readonly=True,
    )
    ai_active_provider_last_sync = fields.Datetime(
        related='ai_active_provider_id.last_sync_date',
        string='Last Models Discovery',
        readonly=True,
    )
    ai_default_model_id = fields.Many2one(
        related='ai_active_provider_id.default_model_id',
        string='Default AI Model',
        readonly=False,
        help='Model used in place of Odoo native OpenAI and Google models.',
    )
    ai_active_provider_embedding_model = fields.Char(
        related='ai_active_provider_id.embedding_model',
        string='Embedding Model',
        readonly=False,
        help='Embedding model used by Odoo native AI sources and RAG.',
    )

    # Use Case Routing Models
    ai_discuss_assistant_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Discuss Agent Model',
        config_parameter='odoo_custom_llm_provider.discuss_assistant_model_id',
        help='Model driving the conversational AI Agent in Discuss and channels.',
    )
    ai_livechat_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Livechat Suggestions Model',
        config_parameter='odoo_custom_llm_provider.livechat_model_id',
        help='Model generating real-time suggestions and answers for Livechat operators.',
    )
    ai_chatter_summary_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Chatter Summary Model',
        config_parameter='odoo_custom_llm_provider.chatter_summary_model_id',
        help='Model used to summarize chatter threads and long message histories.',
    )
    ai_web_editor_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Web Editor /ai Model',
        config_parameter='odoo_custom_llm_provider.web_editor_model_id',
        help='Model powering the rich-text editor slash command /ai and text generation.',
    )
    ai_document_ocr_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Document OCR & Vision Model',
        domain="[('capability_vision', '=', True)]",
        config_parameter='odoo_custom_llm_provider.document_ocr_model_id',
        help='Multimodal vision model for invoice, receipt and document analysis.',
    )

    # Resiliency & Fallback
    ai_fallback_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Fallback (Cascade) Model',
        config_parameter='odoo_custom_llm_provider.fallback_model_id',
        help='Backup model called automatically if the primary model hits rate limits (429) or timeouts.',
    )

    def action_settings_test_provider(self):
        """Action button inside General Settings to test the active provider."""
        self.ensure_one()
        if not self.ai_active_provider_id:
            raise UserError(_("Please select an Active AI Provider first."))
        return self.ai_active_provider_id.action_test_connection()

    def action_settings_sync_models(self):
        """Action button inside General Settings to dynamically discover models."""
        self.ensure_one()
        if not self.ai_active_provider_id:
            raise UserError(_("Please select an Active AI Provider first."))
        return self.ai_active_provider_id.action_fetch_models()

    def set_values(self):
        for settings in self:
            if settings.use_custom_ai_provider:
                if not settings.ai_active_provider_id:
                    raise UserError(_("Select a custom AI provider before enabling it."))
                if not settings.ai_default_model_id:
                    raise UserError(_("Select a default chat model for the custom provider."))
                if settings.ai_default_model_id.provider_id != settings.ai_active_provider_id:
                    raise UserError(_("The default model must belong to the active provider."))
        return super().set_values()
