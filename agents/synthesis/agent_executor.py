"""AgentExecutor for the synthesis agent.

Step 3 of the SPEC. Streams a fake, templated draft answer sentence-by-
sentence as TextPart chunks (append/last_chunk semantics). Also implements
input-required for real: pauses when the sources look too thin or
contradictory, then resumes from the caller's follow-up reply — the first
agent in this repo to do a full input-required round trip.
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

from agents.synthesis.drafting import (
    ambiguity_prompt,
    build_draft,
    parse_prioritized_index,
)
from common.parts import find_data_part_in_history, get_data_part
from common.text import split_sentences

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.

SLEEP_BETWEEN_CHUNKS_SECONDS = 0.1


class SynthesisAgentExecutor(AgentExecutor):
    """Drafts a fake answer paragraph from a topic + sources, streaming it out."""

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
        await self._draft_and_stream(updater, topic, sources)
        await updater.complete()

    async def _resume(
        self, task: Task, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        payload = find_data_part_in_history(task.history or []) or {}
        topic = payload.get("topic", "")
        sources = payload.get("sources", [])
        guidance = context.get_user_input().strip()

        prioritized_index = parse_prioritized_index(guidance, sources)
        present_both = prioritized_index is None and "both" in guidance.lower()
        logger.info(
            "synthesis: resuming task %s with guidance=%r "
            "(prioritized_index=%s, present_both=%s)",
            task.id,
            guidance,
            prioritized_index,
            present_both,
        )

        await updater.start_work()
        await self._draft_and_stream(
            updater,
            topic,
            sources,
            prioritized_index=prioritized_index,
            present_both=present_both,
        )
        await updater.complete()

    async def _draft_and_stream(
        self,
        updater: TaskUpdater,
        topic: str,
        sources: list[dict],
        prioritized_index: int | None = None,
        present_both: bool = False,
    ) -> None:
        draft = build_draft(
            topic,
            sources,
            prioritized_index=prioritized_index,
            present_both=present_both,
        )
        sentences = split_sentences(draft)
        artifact_id = str(uuid4())
        last_index = len(sentences) - 1
        for index, sentence in enumerate(sentences):
            await asyncio.sleep(SLEEP_BETWEEN_CHUNKS_SECONDS)
            chunk_text = sentence if index == 0 else f" {sentence}"
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=chunk_text))],
                artifact_id=artifact_id,
                name="draft",
                append=index > 0,
                last_chunk=index == last_index,
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
