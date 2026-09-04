"""Pure conversion helpers between Odoo AI and provider adapter payloads."""

import base64
import json


CUSTOM_MODEL_PREFIX = "custom_"


def custom_record_id(value):
    """Return the database id encoded in ``custom_<id>`` or ``None``."""
    value = str(value or "")
    if not value.startswith(CUSTOM_MODEL_PREFIX):
        return None
    try:
        return int(value[len(CUSTOM_MODEL_PREFIX):])
    except (TypeError, ValueError):
        return None


def format_tools(tools):
    """Convert Odoo's tool registry to the common OpenAI function schema."""
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": definition[0] or "",
                "parameters": definition[3] or {"type": "object", "properties": {}},
            },
        }
        for name, definition in tools.items()
    ]


def _file_content(file_info):
    mimetype = file_info.get("mimetype") or "application/octet-stream"
    value = file_info.get("value") or ""
    if mimetype == "text/plain":
        return {"type": "text", "text": value}
    if mimetype.startswith("image/"):
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mimetype};base64,{value}"},
        }
    if mimetype == "application/pdf":
        try:
            decoded = base64.b64decode(value).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            decoded = "[PDF attachment supplied; this provider adapter cannot extract its text]"
        return {"type": "text", "text": decoded}
    return {"type": "text", "text": f"[Attached file: {mimetype}]"}


def build_messages(system_prompts, user_prompts, inputs=(), files=None):
    """Build chat-completions messages while preserving tool-call history."""
    messages = []
    system_text = "\n\n".join(str(prompt) for prompt in (system_prompts or ()) if prompt)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for item in inputs or ():
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call_output":
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id"),
                "content": str(item.get("output", "")),
            })
        elif item_type == "function_call":
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": item.get("call_id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "{}",
                    },
                }],
            })
        elif item.get("role") in ("user", "assistant", "tool", "model"):
            messages.append(dict(item))

    prompt_text = "\n\n".join(str(prompt) for prompt in (user_prompts or ()) if prompt)
    file_parts = [_file_content(file_info) for file_info in (files or ())]
    if file_parts:
        content = ([{"type": "text", "text": prompt_text}] if prompt_text else []) + file_parts
        messages.append({"role": "user", "content": content})
    elif prompt_text:
        messages.append({"role": "user", "content": prompt_text})

    return messages


def response_format_for_schema(schema):
    if not schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "odoo_response",
            "schema": schema,
            "strict": True,
        },
    }


def parse_tool_calls(tool_calls):
    """Return Odoo's ``(name, call_id, arguments)`` representation."""
    parsed = []
    for index, tool_call in enumerate(tool_calls or ()):
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name")
        call_id = tool_call.get("id") or name or f"call_{index}"
        arguments = function.get("arguments", tool_call.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {}
        parsed.append((name, call_id, arguments or {}))
    return parsed


def tool_call_history(provider_type, content, tool_calls, provider_history=()):
    """Return the assistant/model history Odoo must replay after tool calls.

    Gemini response parts can contain opaque ``thoughtSignature`` values.  Its
    stateless REST API requires those parts to be sent back unchanged, so use
    the provider-native history whenever the adapter supplies it.  Other
    providers continue to use the normalized common representation.
    """
    if provider_type == "gemini" and provider_history:
        return [dict(item) for item in provider_history]
    return [{
        "role": "assistant",
        "content": content or None,
        "tool_calls": tool_calls,
    }]


def tool_result_message(provider_type, tool_call_id, return_value):
    value = str(return_value)
    if provider_type == "anthropic":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": value,
            }],
        }
    if provider_type == "gemini":
        return {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": tool_call_id,
                    "response": {"result": value},
                },
            }],
        }
    return {"role": "tool", "tool_call_id": tool_call_id, "content": value}
