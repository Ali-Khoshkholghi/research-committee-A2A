"""AgentExecutor for the lit_search agent.

Streams sources for the `find-sources` skill via a pluggable
SearchProvider (agents/lit_search/search_provider.py) — the fake
generator from Step 1, or real web search as of Step 5. The streaming/
task logic here doesn't know or care which provider is behind it, and
the A2A contract (Agent Card, artifact shape, event sequencing) is
identical either way: one TaskArtifactUpdateEvent per source, ending
with last_chunk=True and task state completed.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agents.lit_search.search_provider import (
    DEFAULT_MAX_RESULTS,
    SearchProvider,
    SearchProviderError,
)

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.

SLEEP_BETWEEN_SOURCES_SECONDS = 0.5


class LitSearchAgentExecutor(AgentExecutor):
    """Streams search results for the `find-sources` skill."""

    def __init__(
        self, provider: SearchProvider, max_results: int = DEFAULT_MAX_RESULTS
    ) -> None:
        self._provider = provider
        self._max_results = max_results

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        topic = context.get_user_input().strip()
        if not topic:
            await updater.failed(
                message=new_agent_text_message(
                    "No search topic was provided.", task.context_id, task.id
                )
            )
            return

        logger.info("lit_search: searching for %r", topic)
        try:
            sources = await self._provider.search(topic, self._max_results)
        except SearchProviderError as exc:
            logger.error("lit_search: search failed: %s", exc)
            await updater.failed(
                message=new_agent_text_message(
                    f"Search failed: {exc}", task.context_id, task.id
                )
            )
            return

        # Stream results out one at a time regardless of whether the
        # provider returned them incrementally or as one batch — this is
        # what keeps the streaming contract identical across providers.
        artifact_id = str(uuid4())
        last_index = len(sources) - 1
        for index, source in enumerate(sources):
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
