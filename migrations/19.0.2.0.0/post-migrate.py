# -*- coding: utf-8 -*-
"""Preserve useful configuration created by the pre-native-bridge version."""


NATIVE_MODELS = (
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4o',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-5',
    'gpt-5-mini',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-1.5-pro',
    'gemini-1.5-flash',
)


def migrate(cr, version):
    # Existing installations already have synchronized model catalogs. Pick a
    # deterministic default so enabling the global override never forwards a
    # native model name to an unrelated endpoint.
    cr.execute("""
        UPDATE ai_provider AS provider
           SET default_model_id = (
               SELECT model.id
                 FROM ai_model AS model
                WHERE model.provider_id = provider.id
                  AND model.active
                  AND model.capability_chat
             ORDER BY model.sequence, model.id
                LIMIT 1
           )
         WHERE provider.default_model_id IS NULL
    """)

    # The previous version also stored technical model names directly in
    # ai.agent. Convert only non-native values; new values include the record id
    # and therefore cannot collide across providers.
    cr.execute("""
        WITH candidate AS (
            SELECT DISTINCT ON (agent.id)
                   agent.id AS agent_id,
                   model.id AS model_id
              FROM ai_agent AS agent
              JOIN ai_model AS model ON model.model_id = agent.llm_model
              JOIN ai_provider AS provider ON provider.id = model.provider_id
             WHERE agent.llm_model NOT LIKE 'custom_%%'
               AND agent.llm_model NOT IN %s
               AND model.active
               AND provider.active
          ORDER BY agent.id, provider.sequence, model.sequence, model.id
        )
        UPDATE ai_agent AS agent
           SET llm_model = 'custom_' || candidate.model_id::varchar
          FROM candidate
         WHERE agent.id = candidate.agent_id
    """, (NATIVE_MODELS,))
