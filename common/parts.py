"""Helpers for pulling typed Part content out of an A2A Message."""

from __future__ import annotations

from typing import Any

from a2a.types import DataPart, Message


def get_data_part(message: Message | None) -> dict[str, Any] | None:
    """Returns the `data` dict of the first DataPart in a message, if any."""
    if message is None:
        return None
    for part in message.parts:
        if isinstance(part.root, DataPart):
            return part.root.data
    return None


def find_data_part_in_history(messages: list[Message]) -> dict[str, Any] | None:
    """Returns the `data` dict of the first DataPart found across messages.

    Used to recover the original request payload after an input-required
    pause, where the current message is the caller's follow-up reply and
    the original DataPart lives earlier in the task's history.
    """
    for message in messages:
        data = get_data_part(message)
        if data is not None:
            return data
    return None
