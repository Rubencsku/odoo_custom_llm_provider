# -*- coding: utf-8 -*-
import json
import logging
from odoo import models, fields, api, _
from odoo.exceptions import UserError, ValidationError

from ..services import (
    get_adapter,
    AIProviderError,
    AIAuthenticationError,
    AIRateLimitError,
    AITimeoutError,
)

_logger = logging.getLogger(__name__)

PROVIDER_DEFAULT_URLS = {
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com/v1',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta',
    'groq': 'https://api.groq.com/openai/v1',
    'openrouter': 'https://openrouter.ai/api/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'ollama': 'http://localhost:11434',
    'custom_openai': 'http://localhost:8000/v1',
}


class AIProvider(models.Model):
    _name = 'ai.provider'
    _description = 'AI / LLM Provider'
    _order = 'sequence, name'

    name = fields.Char(
        string='Provider Name',
        required=True,
        help='Descriptive name for this provider (e.g. OpenAI Cloud, Anthropic Claude, Ollama Server).',
    )
    sequence = fields.Integer(string='Sequence', default=10)
    active = fields.Boolean(string='Active', default=True)
    is_default = fields.Boolean(
        string='Default Provider',
        default=False,
        help='Designate this provider as the system default.',
    )

    provider_type = fields.Selection(
        selection=[
            ('openai', 'OpenAI'),
            ('anthropic', 'Anthropic Claude'),
            ('gemini', 'Google Gemini'),
            ('groq', 'Groq (Ultra-fast)'),
            ('openrouter', 'OpenRouter'),
            ('deepseek', 'DeepSeek'),
            ('ollama', 'Ollama (Local / On-Premise)'),
            ('custom_openai', 'Custom OpenAI-Compatible (vLLM / LM Studio / etc.)'),
        ],
        string='Provider Type',
        required=True,
        default='openai',
        help='Protocol and format used to communicate with this provider.',
    )

    base_url = fields.Char(
        string='API Base URL',
        required=True,
        default='https://api.openai.com/v1',
        help='Endpoint base URL (e.g. https://api.openai.com/v1 or http://localhost:11434).',
    )

    api_key = fields.Char(
        string='API Key',
        groups='base.group_system',
        help='Secret API authentication key. Kept encrypted and hidden from non-admin users.',
    )

    timeout = fields.Integer(
        string='Timeout (Seconds)',
        default=60,
        required=True,
        help='Max duration to wait for LLM completion before timing out or attempting fallback.',
    )

    custom_headers = fields.Text(
        string='Custom HTTP Headers (JSON)',
        help='Optional JSON key-value pairs of extra headers (e.g. {"HTTP-Referer": "https://mycompany.com"}).',
    )

    embedding_model = fields.Char(
        string='Embedding Model',
        compute='_compute_embedding_model',
        inverse='_inverse_embedding_model',
        search='_search_embedding_model',
        store=False,
        help='Model used for RAG vector embeddings (e.g. text-embedding-3-small, nomic-embed-text).',
    )

    # Models & Logs
    model_ids = fields.One2many(
        comodel_name='ai.model',
        inverse_name='provider_id',
        string='Models Catalog',
    )
    default_model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Default Chat Model',
        domain="[('provider_id', '=', id), ('active', '=', True), ('capability_chat', '=', True)]",
        help=(
            'Model used when this provider replaces an Odoo native provider. '
            'Agents that explicitly select another custom model keep using that model.'
        ),
    )
    model_count = fields.Integer(
        string='Models Count',
        compute='_compute_model_count',
        store=False,
    )

    log_ids = fields.One2many(
        comodel_name='ai.request.log',
        inverse_name='provider_id',
        string='Request Logs',
    )
    log_count = fields.Integer(
        string='Logs Count',
        compute='_compute_log_count',
        store=False,
    )

    # Diagnostics
    status = fields.Selection(
        selection=[
            ('draft', 'Not Tested'),
            ('connected', 'Connected'),
            ('error', 'Error / Unreachable'),
        ],
        string='Connection Status',
        default='draft',
        copy=False,
    )
    status_message = fields.Text(string='Diagnostic Message', copy=False, readonly=True)
    last_sync_date = fields.Datetime(string='Last Models Sync', copy=False, readonly=True)

    @api.onchange('provider_type')
    def _onchange_provider_type(self):
        if self.provider_type and self.provider_type in PROVIDER_DEFAULT_URLS:
            self.base_url = PROVIDER_DEFAULT_URLS[self.provider_type]

    @api.constrains('default_model_id')
    def _check_default_model_provider(self):
        for rec in self:
            if rec.default_model_id and rec.default_model_id.provider_id != rec:
                raise ValidationError(_('The default model must belong to this provider.'))

    def _compute_model_count(self):
        for rec in self:
            rec.model_count = len(rec.model_ids)

    def _compute_log_count(self):
        for rec in self:
            rec.log_count = len(rec.log_ids)

    def _compute_embedding_model(self):
        Param = self.env['ir.config_parameter'].sudo()
        for rec in self:
            val = Param.get_param(f'odoo_custom_llm_provider.provider_{rec.id}_embedding_model')
            if not val:
                # Provide reasonable default based on provider type
                if rec.provider_type == 'ollama':
                    val = 'nomic-embed-text'
                elif rec.provider_type == 'gemini':
                    val = 'gemini-embedding-001'
                else:
                    val = 'text-embedding-3-small'
            rec.embedding_model = val

    def _inverse_embedding_model(self):
        Param = self.env['ir.config_parameter'].sudo()
        for rec in self:
            if rec.embedding_model:
                Param.set_param(
                    f'odoo_custom_llm_provider.provider_{rec.id}_embedding_model',
                    str(rec.embedding_model).strip()
                )

    def _search_embedding_model(self, operator, value):
        prefix = 'odoo_custom_llm_provider.provider_'
        suffix = '_embedding_model'
        params = self.env['ir.config_parameter'].sudo().search([
            ('key', '=like', f'{prefix}%{suffix}'),
            ('value', operator, value)
        ])
        provider_ids = []
        for p in params:
            try:
                provider_ids.append(int(p.key[len(prefix):-len(suffix)]))
            except (TypeError, ValueError):
                pass
        return [('id', 'in', provider_ids)]

    def _parse_custom_headers(self):
        if not self.custom_headers:
            return {}
        try:
            parsed = json.loads(self.custom_headers)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception as e:
            _logger.warning("Failed parsing custom_headers for provider %s: %s", self.name, e)
        return {}

    def get_adapter(self):
        """Instantiate and return the configured services adapter."""
        self.ensure_one()
        headers = self._parse_custom_headers()
        return get_adapter(
            provider_type=self.provider_type,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            custom_headers=headers,
        )

    def action_test_connection(self):
        """Test live endpoint and API key. Update diagnostics and notify user."""
        self.ensure_one()
        adapter = self.get_adapter()
        success, message = adapter.test_connection()

        self.status = 'connected' if success else 'error'
        self.status_message = message

        notification_type = 'success' if success else 'danger'
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Connection Test Result'),
                'message': message,
                'type': notification_type,
                'sticky': not success,
            },
        }

    def action_fetch_models(self):
        """
        Dynamically query provider endpoint to discover all available models,
        creating new ai.model records or updating existing ones.
        """
        self.ensure_one()
        adapter = self.get_adapter()

        try:
            discovered_models = adapter.fetch_available_models()
        except AIProviderError as e:
            self.status = 'error'
            self.status_message = str(e)
            raise UserError(_('Failed to fetch models from provider: %s') % str(e))
        except Exception as e:
            self.status = 'error'
            self.status_message = str(e)
            raise UserError(_('Unexpected error querying provider models: %s') % str(e))

        if not discovered_models:
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _('Model Discovery'),
                    'message': _('No models were returned by the provider endpoint.'),
                    'type': 'warning',
                    'sticky': False,
                },
            }

        AIModel = self.env['ai.model']
        created_count = 0
        updated_count = 0

        for m_info in discovered_models:
            existing = AIModel.search([
                ('provider_id', '=', self.id),
                ('model_id', '=', m_info.model_id),
            ], limit=1)

            vals = {
                'name': m_info.name,
                'model_id': m_info.model_id,
                'provider_id': self.id,
                'capability_chat': m_info.capability_chat,
                'capability_vision': m_info.capability_vision,
                'capability_tools': m_info.capability_tools,
                'capability_json': m_info.capability_json,
                'capability_embeddings': m_info.capability_embeddings,
                'description': m_info.description,
            }

            if existing:
                existing.write(vals)
                updated_count += 1
            else:
                AIModel.create(vals)
                created_count += 1

        if not self.default_model_id:
            self.default_model_id = AIModel.search([
                ('provider_id', '=', self.id),
                ('active', '=', True),
                ('capability_chat', '=', True),
            ], order='sequence, id', limit=1)

        for m_info in discovered_models:
            if m_info.capability_embeddings and (not self.embedding_model or self.embedding_model == 'text-embedding-3-small'):
                self.embedding_model = m_info.model_id
                break

        self.last_sync_date = fields.Datetime.now()
        self.status = 'connected'
        self.status_message = _('Successfully synced %d models (%d added, %d updated).') % (
            len(discovered_models), created_count, updated_count
        )

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': _('Model Sync Complete'),
                'message': _('Successfully synchronized %d models: %d created, %d updated.') % (
                    len(discovered_models), created_count, updated_count
                ),
                'type': 'success',
                'sticky': False,
            },
        }

    def action_view_models(self):
        self.ensure_one()
        return {
            'name': _('Models: %s') % self.name,
            'type': 'ir.actions.act_window',
            'res_model': 'ai.model',
            'view_mode': 'list,form',
            'domain': [('provider_id', '=', self.id)],
            'context': {'default_provider_id': self.id},
        }

    def action_view_logs(self):
        self.ensure_one()
        return {
            'name': _('Logs: %s') % self.name,
            'type': 'ir.actions.act_window',
            'res_model': 'ai.request.log',
            'view_mode': 'list,form,pivot,graph',
            'domain': [('provider_id', '=', self.id)],
            'context': {'default_provider_id': self.id},
        }
