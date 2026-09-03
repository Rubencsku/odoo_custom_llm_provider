# -*- coding: utf-8 -*-
import os
import sys
import unittest
import xml.etree.ElementTree as ET


MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if MODULE_PATH not in sys.path:
    sys.path.insert(0, MODULE_PATH)

from services.native_bridge import (
    build_messages,
    custom_record_id,
    format_tools,
    parse_tool_calls,
    response_format_for_schema,
    tool_result_message,
)


class TestNativeAIIntegration(unittest.TestCase):

    def test_custom_ids_are_unambiguous(self):
        self.assertEqual(custom_record_id('custom_101'), 101)
        self.assertIsNone(custom_record_id('llama3.2'))
        self.assertIsNone(custom_record_id('custom_invalid'))

        native_selection = [('gpt-4o', 'GPT-4o')]
        custom_selection = native_selection + [('custom_101', 'Llama 3.2 [Ollama]')]
        self.assertNotIn(('llama3.2', 'Llama 3.2 [Ollama]'), custom_selection)

    def test_odoo_prompts_and_history_become_chat_messages(self):
        messages = build_messages(
            ['System A', 'System B'],
            ['New prompt'],
            inputs=[
                {'role': 'user', 'content': 'Earlier question'},
                {
                    'type': 'function_call',
                    'call_id': 'call_1',
                    'name': 'lookup_partner',
                    'arguments': '{"id": 7}',
                },
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': 'Ruben',
                },
            ],
        )
        self.assertEqual(messages[0], {'role': 'system', 'content': 'System A\n\nSystem B'})
        self.assertEqual(messages[1], {'role': 'user', 'content': 'Earlier question'})
        self.assertEqual(messages[2]['tool_calls'][0]['function']['name'], 'lookup_partner')
        self.assertEqual(messages[3]['role'], 'tool')
        self.assertEqual(messages[-1], {'role': 'user', 'content': 'New prompt'})

    def test_tool_schema_and_response_conversion(self):
        tools = {
            'lookup_partner': (
                'Look up a partner',
                False,
                object(),
                {
                    'type': 'object',
                    'properties': {'id': {'type': 'integer'}},
                    'required': ['id'],
                },
            ),
        }
        formatted = format_tools(tools)
        self.assertEqual(formatted[0]['function']['name'], 'lookup_partner')

        parsed = parse_tool_calls([{
            'id': 'call_1',
            'type': 'function',
            'function': {'name': 'lookup_partner', 'arguments': '{"id": 7}'},
        }])
        self.assertEqual(parsed, [('lookup_partner', 'call_1', {'id': 7})])
        self.assertEqual(tool_result_message('openai', 'call_1', 'ok')['role'], 'tool')
        self.assertEqual(
            tool_result_message('anthropic', 'call_1', 'ok')['content'][0]['type'],
            'tool_result',
        )
        self.assertIn('functionResponse', tool_result_message('gemini', 'lookup_partner', 'ok')['parts'][0])

    def test_json_schema_uses_chat_completions_shape(self):
        schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
        response_format = response_format_for_schema(schema)
        self.assertEqual(response_format['type'], 'json_schema')
        self.assertEqual(response_format['json_schema']['schema'], schema)

    def test_settings_extend_native_ai_app_view(self):
        root = ET.parse(os.path.join(MODULE_PATH, 'views', 'res_config_settings_views.xml')).getroot()
        inherit_field = root.find(".//record[@id='res_config_settings_view_form_ai_custom']/field[@name='inherit_id']")
        self.assertEqual(inherit_field.attrib['ref'], 'ai_app.res_config_settings_view_form')
        self.assertIsNotNone(root.find(".//setting[@id='custom_ai_provider']"))


if __name__ == '__main__':
    unittest.main()
