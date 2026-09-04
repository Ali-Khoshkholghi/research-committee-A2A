"""AgentExecutor for the synthesis agent.

Streams a drafted answer for the `draft-answer` skill via a pluggable
SynthesisProvider (agents/synthesis/llm_provider.py) — the naive
templater from Step 3, or a real Gemini call as of Step 6. The
streaming/task logic here doesn't know or care which provider is behind
it: it consumes whatever text chunks the provider yields and streams
them out as TextPart artifacts with the same append/last_chunk
semantics either way.

Also implements input-required for real: pauses when the sources look
too thin or contradictory (this check runs before any provider is
called, regardless of which one is selected), then resumes from the
caller's follow-up reply.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, Task, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agents.synthesis.drafting import ambiguity_prompt
from agents.synthesis.llm_provider import SynthesisProvider, SynthesisProviderError
from common.parts import find_data_part_in_history, get_data_part

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.

SLEEP_BETWEEN_CHUNKS_SECONDS = 0.1


class SynthesisAgentExecutor(AgentExecutor):
    """Drafts an answer paragraph from a topic + sources, streaming it out."""

    def __init__(self, provider: SynthesisProvider) -> None:
        self._provider = provider

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            await self._start(context, event_queue)
        elif task.status.state == TaskState.input_required:
            await self._resume(task, context, event_queue)
        else:
            logger.warning(
                "synthesis: unexpected re-invocation of task %s in state %s",
                task.id,
                task.status.state,
            )

    async def _start(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        payload = get_data_part(context.message) or {}
        topic = payload.get("topic", "")
        sources = payload.get("sources", [])

        prompt = ambiguity_prompt(sources)
        if prompt is not None:
            logger.info("synthesis: pausing for input — %s", prompt)
            await updater.requires_input(
                message=new_agent_text_message(prompt, task.context_id, task.id)
            )
            return

        await updater.start_work()
        try:
            await self._draft_and_stream(updater, task, topic, sources)
        except SynthesisProviderError as exc:
            await self._fail(updater, task, exc)
            return
        await updater.complete()

    async def _resume(
        self, task: Task, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        payload = find_data_part_in_history(task.history or []) or {}
        topic = payload.get("topic", "")
        sources = payload.get("sources", [])
        guidance = context.get_user_input().strip()
        logger.info("synthesis: resuming task %s with guidance=%r", task.id, guidance)

        await updater.start_work()
        try:
            await self._draft_and_stream(updater, task, topic, sources, guidance=guidance)
        except SynthesisProviderError as exc:
            await self._fail(updater, task, exc)
            return
        await updater.complete()

    async def _draft_and_stream(
        self,
        updater: TaskUpdater,
        task: Task,
        topic: str,
        sources: list[dict],
        guidance: str | None = None,
    ) -> None:
        """Streams the provider's chunks out as TextPart artifacts.

        The provider yields chunks lazily and we don't know which one is
        last until the iterator is exhausted, so this buffers one chunk
        ahead to set last_chunk correctly on the true final event.
        """
        artifact_id = str(uuid4())
        chunks = self._provider.draft(topic, sources, guidance).__aiter__()

        try:
            current = await chunks.__anext__()
        except StopAsyncIteration as exc:
            raise SynthesisProviderError("provider returned no text") from exc

        index = 0
        while True:
            try:
                next_chunk = await chunks.__anext__()
            except StopAsyncIteration:
                await asyncio.sleep(SLEEP_BETWEEN_CHUNKS_SECONDS)
                await updater.add_artifact(
                    parts=[Part(root=TextPart(text=current))],
                    artifact_id=artifact_id,
                    name="draft",
                    append=index > 0,
                    last_chunk=True,
                )
                return

            await asyncio.sleep(SLEEP_BETWEEN_CHUNKS_SECONDS)
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=current))],
                artifact_id=artifact_id,
                name="draft",
                append=index > 0,
                last_chunk=False,
            )
            current = next_chunk
            index += 1

    async def _fail(
        self, updater: TaskUpdater, task: Task, exc: SynthesisProviderError
    ) -> None:
        logger.error("synthesis: %s", exc)
        await updater.failed(
            message=new_agent_text_message(
                f"Draft failed: {exc}", task.context_id, task.id
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
