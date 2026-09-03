# -*- coding: utf-8 -*-
import logging
from odoo import models, api, _

_logger = logging.getLogger(__name__)


class DiscussChannel(models.Model):
    _inherit = 'discuss.channel'

    def execute_command_ai(self, **kwargs):
        """
        Hook into Odoo Discuss /ai slash command.
        If custom AI provider is active, routes directly to our configured LLM.
        """
        service = self.env['ai.service']
        if not service.is_custom_ai_enabled():
            if hasattr(super(), 'execute_command_ai'):
                return super().execute_command_ai(**kwargs)
            return False

        prompt = kwargs.get('body') or kwargs.get('content') or ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the official AI Assistant embedded inside Odoo Discuss. "
                    "Provide clear, professional, concise answers to help the team."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            resp = service.execute_chat(
                use_case='discuss_assistant',
                messages=messages,
            )
            # Post reply into channel
            self.message_post(
                body=resp.content,
                message_type='comment',
                subtype_xmlid='mail.mt_comment',
            )
            return True
        except Exception as e:
            _logger.error("Error in Discuss /ai command: %s", e)
            self.message_post(
                body=_("⚠️ AI Assistant Error: %s") % str(e),
                message_type='comment',
            )
            return False

    def action_summarize_conversation(self):
        """Summarize channel or thread conversation using chatter_summary model."""
        self.ensure_one()
        service = self.env['ai.service']
        recent_messages = self.message_ids.sorted('create_date', reverse=False)[-20:]
        transcript = []
        for m in recent_messages:
            author = m.author_id.name or _("User")
            clean_body = m.body or ""
            transcript.append(f"{author}: {clean_body}")

        messages = [
            {
                "role": "system",
                "content": "You are an executive assistant in Odoo. Provide a brief bulleted summary of this conversation with action items.",
            },
            {"role": "user", "content": "\n".join(transcript)},
        ]

        resp = service.execute_chat(
            use_case='chatter_summary',
            messages=messages,
        )
        return resp.content


class MailThread(models.AbstractModel):
    _inherit = 'mail.thread'

    def action_summarize_chatter(self):
        """
        Summarize the chatter history of any document (CRM Lead, Sale Order, Helpdesk Ticket, etc.).
        """
        self.ensure_one()
        service = self.env['ai.service']
        if not service.is_custom_ai_enabled():
            if hasattr(super(), 'action_summarize_chatter'):
                return super().action_summarize_chatter()

        messages_history = self.message_ids.sorted('create_date', reverse=False)[-15:]
        history_text = []
        for m in messages_history:
            sender = m.author_id.name or _("Unknown")
            history_text.append(f"[{m.date}] {sender}: {m.body or ''}")

        prompt = (
            f"Please generate a concise 3-5 bullet point executive summary of the following document history "
            f"for record '{self.display_name}' (Model: {self._name}):\n\n"
            + "\n".join(history_text)
        )

        response = service.execute_chat(
            use_case='chatter_summary',
            messages=[
                {"role": "system", "content": "You are a concise enterprise document assistant in Odoo."},
                {"role": "user", "content": prompt},
            ]
        )

        summary_body = (
            f"<div class='alert alert-info' role='alert'>"
            f"<strong><i class='fa fa-magic'></i> AI Executive Summary:</strong><br/>"
            f"{response.content}"
            f"</div>"
        )
        self.message_post(body=summary_body, message_type='notification')
        return True
