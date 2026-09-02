"""AgentExecutor for the lit_search agent.

Step 1 of the SPEC: prove the A2A streaming/task/artifact mechanics work
end-to-end with fake data. No real web search yet (that's Step 5).
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, UnsupportedOperationError
from a2a.utils import new_task
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.

# Fake source generator standing in for a real web search (Step 5).
FAKE_SOURCES = [
    {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/abs/1706.03762",
        "snippet": (
            "Introduces the Transformer architecture, replacing recurrence "
            "and convolutions with self-attention."
        ),
    },
    {
        "title": "Deep Residual Learning for Image Recognition",
        "url": "https://arxiv.org/abs/1512.03385",
        "snippet": (
            "Proposes residual connections that make very deep "
            "convolutional networks trainable."
        ),
    },
    {
        "title": "Language Models are Few-Shot Learners",
        "url": "https://arxiv.org/abs/2005.14165",
        "snippet": (
            "Shows that scaling autoregressive language models yields "
            "strong few-shot task performance without fine-tuning."
        ),
    },
    {
        "title": "A Survey of Large Language Models",
        "url": "https://arxiv.org/abs/2303.18223",
        "snippet": (
            "Surveys the recent development of large language models, "
            "covering pre-training, adaptation, and evaluation."
        ),
    },
]

SLEEP_BETWEEN_SOURCES_SECONDS = 0.5


class LitSearchAgentExecutor(AgentExecutor):
    """Streams a handful of fake sources for the `find-sources` skill."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        topic = context.get_user_input() or "(no topic given)"
        logger.info("lit_search: searching for %r", topic)

        artifact_id = str(uuid4())
        last_index = len(FAKE_SOURCES) - 1
        for index, source in enumerate(FAKE_SOURCES):
            await asyncio.sleep(SLEEP_BETWEEN_SOURCES_SECONDS)
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=source))],
                artifact_id=artifact_id,
                name="sources",
                append=index > 0,
                last_chunk=index == last_index,
            )

        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
