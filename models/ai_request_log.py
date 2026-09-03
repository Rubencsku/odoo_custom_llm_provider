# -*- coding: utf-8 -*-
from odoo import models, fields, api, _


class AIRequestLog(models.Model):
    _name = 'ai.request.log'
    _description = 'AI Request & Token Audit Log'
    _order = 'create_date desc, id desc'

    name = fields.Char(string='Log Reference', required=True, copy=False, readonly=True, default=lambda self: _('New'))
    user_id = fields.Many2one(
        comodel_name='res.users',
        string='User',
        default=lambda self: self.env.user,
        index=True,
        readonly=True,
    )
    provider_id = fields.Many2one(
        comodel_name='ai.provider',
        string='Provider',
        index=True,
        readonly=True,
    )
    model_id = fields.Many2one(
        comodel_name='ai.model',
        string='Model',
        index=True,
        readonly=True,
    )
    use_case = fields.Selection(
        selection=[
            ('discuss_assistant', 'Discuss AI Agent / Chat'),
            ('livechat', 'Livechat AI Suggestions'),
            ('chatter_summary', 'Chatter Thread Summary'),
            ('web_editor', 'Web Editor (/ai Assistant)'),
            ('document_ocr', 'Document OCR & Extraction'),
            ('custom', 'Custom Service / Mixin Call'),
        ],
        string='Use Case / Feature',
        index=True,
        readonly=True,
        default='custom',
    )
    status = fields.Selection(
        selection=[
            ('success', 'Success'),
            ('fallback_used', 'Fallback Succeeded'),
            ('error', 'Failed / Error'),
        ],
        string='Status',
        default='success',
        index=True,
        readonly=True,
    )

    # Token Accounting & Performance
    prompt_tokens = fields.Integer(string='Prompt Tokens', readonly=True)
    completion_tokens = fields.Integer(string='Completion Tokens', readonly=True)
    total_tokens = fields.Integer(string='Total Tokens', readonly=True)
    latency_ms = fields.Float(string='Latency (ms)', digits=(10, 2), readonly=True)
    cost_estimated = fields.Float(string='Est. Cost ($)', digits=(10, 6), readonly=True)

    # Snippets & Errors (Limited length for privacy and performance)
    prompt_snippet = fields.Text(string='Prompt Preview', readonly=True)
    response_snippet = fields.Text(string='Response Preview', readonly=True)
    error_message = fields.Text(string='Error Details', readonly=True)

    @api.model_create_multi
    def create(self, vals_list):
        for vals in vals_list:
            if vals.get('name', _('New')) == _('New'):
                vals['name'] = self.env['ir.sequence'].next_by_code('ai.request.log') or f"LOG-{fields.Datetime.now().strftime('%Y%m%d%H%M%S')}"
        return super().create(vals_list)
