"""Helpers for building outbound A2A messages to sub-agents."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from a2a.types import DataPart, Message, Part, Role, TextPart


def text_message(
    text: str, context_id: str, task_id: str | None = None
) -> Message:
    return Message(
        role=Role.user,
        message_id=str(uuid4()),
        context_id=context_id,
        task_id=task_id,
        parts=[Part(root=TextPart(text=text))],
    )


def data_message(
    data: dict[str, Any], context_id: str, task_id: str | None = None
) -> Message:
    return Message(
        role=Role.user,
        message_id=str(uuid4()),
        context_id=context_id,
        task_id=task_id,
        parts=[Part(root=DataPart(data=data))],
    )
