# -*- coding: utf-8 -*-
"""Allow Odoo's native RAG records to store custom embedding model names."""

from odoo import api, fields, models

from odoo.addons.ai.utils.llm_providers import EMBEDDING_MODELS_SELECTION


class AIEmbedding(models.Model):
    _inherit = 'ai.embedding'

    embedding_model = fields.Selection(
        selection='_get_embedding_model_selection',
        string='Embedding Model',
        required=True,
    )

    @api.model
    def _get_embedding_model_selection(self):
        selection = list(EMBEDDING_MODELS_SELECTION)
        existing = {value for value, _label in selection}
        providers = self.env['ai.provider'].sudo().search([('active', '=', True)])
        for provider in providers:
            value = provider.embedding_model
            if value and value not in existing:
                selection.append((value, f'{value} [{provider.name}]'))
                existing.add(value)
        return selection
