# -*- coding: utf-8 -*-
{
    'name': 'Odoo AI Custom Provider',
    'version': '19.0.2.0.1',
    'category': 'Productivity/Artificial Intelligence',
    'summary': 'Use custom LLM providers from Odoo 19 native AI settings and agents',
    'description': """
Native custom-provider support for Odoo 19 AI
==============================================
Configure a custom LLM from Odoo's own AI settings and use it with native
agents, Discuss, tools and RAG. Supports OpenAI-compatible APIs, Anthropic,
Google Gemini and Ollama, including model discovery and request diagnostics.
    """,
    'author': 'Ruben Oviedo / Custom AI Provider Team',
    'website': 'https://github.com/Rubencsku/odoo_custom_llm_provider',
    'license': 'LGPL-3',
    'depends': [
        'base',
        'mail',
        'ai',
        'ai_app',
    ],
    'data': [
        'security/security.xml',
        'security/ir.model.access.csv',
        'data/ai_provider_data.xml',
        'views/ai_provider_views.xml',
        'views/ai_model_views.xml',
        'views/ai_request_log_views.xml',
        'views/res_config_settings_views.xml',
        'views/menus.xml',
    ],
    'installable': True,
    'application': False,
    'auto_install': False,
}
