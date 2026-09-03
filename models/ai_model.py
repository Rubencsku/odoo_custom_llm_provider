# -*- coding: utf-8 -*-
import logging
from odoo import models, fields, api, _
from odoo.exceptions import UserError, ValidationError

_logger = logging.getLogger(__name__)


class AIModel(models.Model):
    _name = 'ai.model'
    _description = 'AI / LLM Model'
    _order = 'provider_id, sequence, name'

    name = fields.Char(string='Model Display Name', required=True)
    model_id = fields.Char(
        string='Technical Model ID',
        required=True,
        help='Technical identifier sent to the LLM API (e.g. gpt-4o, claude-3-7-sonnet-20250219, qwen2.5:32b).',
    )
    provider_id = fields.Many2one(
        comodel_name='ai.provider',
        string='Provider',
        required=True,
        ondelete='cascade',
        index=True,
    )
    provider_type = fields.Selection(
        related='provider_id.provider_type',
        string='Provider Type',
        store=True,
        readonly=True,
    )
    active = fields.Boolean(string='Active', default=True)
    sequence = fields.Integer(string='Sequence', default=10)
    description = fields.Text(string='Description')

    # Capabilities
    capability_chat = fields.Boolean(string='Chat / Dialog', default=True)
    capability_vision = fields.Boolean(string='Vision / Multimodal', default=False)
    capability_tools = fields.Boolean(string='Tool / Function Calling', default=True)
    capability_json = fields.Boolean(string='Structured JSON', default=True)
    capability_embeddings = fields.Boolean(string='Embeddings', default=False)

    # Default Inference Hyperparameters
    temperature = fields.Float(string='Temperature', default=0.7, help='Range 0.0 to 2.0 (creativity/randomness).')
    max_tokens = fields.Integer(string='Max Tokens', default=2048, help='Maximum output generation tokens.')
    top_p = fields.Float(string='Top P', default=1.0, help='Nucleus sampling probability.')

    # Cost Estimation (per 1 Million tokens)
    cost_per_1m_input_tokens = fields.Float(
        string='Cost / 1M Input Tokens ($)',
        digits=(10, 4),
        default=0.0,
        help='Cost in USD for 1,000,000 prompt/input tokens.',
    )
    cost_per_1m_output_tokens = fields.Float(
        string='Cost / 1M Output Tokens ($)',
        digits=(10, 4),
        default=0.0,
        help='Cost in USD for 1,000,000 completion/output tokens.',
    )

    # Usage Stats
    log_ids = fields.One2many(
        comodel_name='ai.request.log',
        inverse_name='model_id',
        string='Usage Logs',
    )
    total_tokens_used = fields.Integer(string='Total Tokens Used', compute='_compute_usage_stats', store=False)
    total_requests = fields.Integer(string='Total Requests', compute='_compute_usage_stats', store=False)

    @api.constrains('provider_id', 'model_id')
    def _check_provider_model_unique(self):
        for rec in self:
            domain = [
                ('provider_id', '=', rec.provider_id.id),
                ('model_id', '=', rec.model_id),
                ('id', '!=', rec.id),
            ]
            if self.search_count(domain) > 0:
                raise ValidationError(_("This Model ID already exists for this Provider!"))


    def _compute_usage_stats(self):
        Log = self.env['ai.request.log']
        for rec in self:
            logs = Log.search([('model_id', '=', rec.id)])
            rec.total_requests = len(logs)
            rec.total_tokens_used = sum(logs.mapped('total_tokens'))

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate approximate USD cost for given token count."""
        self.ensure_one()
        cost_in = (prompt_tokens / 1_000_000.0) * (self.cost_per_1m_input_tokens or 0.0)
        cost_out = (completion_tokens / 1_000_000.0) * (self.cost_per_1m_output_tokens or 0.0)
        return round(cost_in + cost_out, 6)

    def generate_chat(self, messages, temperature=None, max_tokens=None, top_p=None, tools=None, response_format=None, **kwargs):
        """
        Execute chat completion through provider adapter.
        """
        self.ensure_one()
        if not self.provider_id.active:
            raise UserError(_("Provider '%s' is inactive.") % self.provider_id.name)

        adapter = self.provider_id.get_adapter()
        temp = self.temperature if temperature is None else temperature
        limit = self.max_tokens if max_tokens is None else max_tokens
        p_val = self.top_p if top_p is None else top_p

        return adapter.chat_completion(
            model_id=self.model_id,
            messages=messages,
            temperature=temp,
            max_tokens=limit,
            top_p=p_val,
            tools=tools,
            response_format=response_format,
            **kwargs
        )

    def action_test_chat(self):
        """Quick diagnostic action: sends a short test prompt and displays the response."""
        self.ensure_one()
        test_messages = [
            {"role": "system", "content": "You are an AI assistant integrated in Odoo 19 Enterprise."},
            {"role": "user", "content": "Respond in one short sentence confirming you are online and working properly."},
        ]
        try:
            response = self.generate_chat(test_messages, max_tokens=60)
            return {
                'type': 'ir.actions.client',
                'tag': 'display_notification',
                'params': {
                    'title': _("Test Success: %s") % self.name,
                    'message': _("Model response:\n%s\n(Latency: %.0f ms, Tokens: %d)") % (
                        response.content, response.latency_ms, response.total_tokens
                    ),
                    'type': 'success',
                    'sticky': False,
                },
            }
        except Exception as e:
            raise UserError(_("Chat test failed: %s") % str(e))
