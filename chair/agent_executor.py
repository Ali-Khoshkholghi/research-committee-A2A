"""AgentExecutor for the chair agent.

Step 4 of the SPEC: the orchestrator. Calls lit_search, synthesis, and
critic over A2A for each human research question, relaying their
progress upward as chair's own status updates, and returns a final
artifact combining synthesis's draft with critic's verdicts.

All three sub-agent calls share one contextId (chair's own task's
context_id) so the whole exchange is traceable as one logical
conversation, even though each hop is a separate A2A Task.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import httpx
from a2a.client.errors import A2AClientError
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    Part,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from chair.client import SubAgentError, build_client, build_client_factory
from chair.messages import data_message, text_message
from common.parts import message_text

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.

# Errors raised while iterating a sub-agent's response stream (network
# failures, malformed responses, etc.) — wrapped into SubAgentError so
# callers only need to handle one exception type.
_TRANSPORT_ERRORS = (A2AClientError, httpx.HTTPError, OSError)


class ChairAgentExecutor(AgentExecutor):
    """Orchestrates lit_search -> synthesis -> critic for one research question."""

    def __init__(self, cards: dict[str, AgentCard]) -> None:
        self._cards = cards
        self._httpx_client: httpx.AsyncClient | None = None
        self._client_factory = None

    def _client_for(self, name: str):
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, read=60.0)
            )
            self._client_factory = build_client_factory(self._httpx_client)
        return build_client(self._client_factory, self._cards[name])

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
                "chair: unexpected re-invocation of task %s in state %s",
                task.id,
                task.status.state,
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())

    # -- top-level flow ----------------------------------------------------

    async def _start(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        topic = context.get_user_input().strip()
        if not topic:
            await updater.failed(
                message=new_agent_text_message(
                    "No research question was provided.", task.context_id, task.id
                )
            )
            return

        shared_context_id = task.context_id

        try:
            sources = await self._run_lit_search(updater, task, shared_context_id, topic)
        except SubAgentError as exc:
            await self._fail(updater, task, exc)
            return

        synthesis_message = data_message(
            {"topic": topic, "sources": sources}, shared_context_id
        )
        try:
            draft, paused = await self._stream_synthesis(
                updater, task, shared_context_id, synthesis_message, topic, sources
            )
        except SubAgentError as exc:
            await self._fail(updater, task, exc)
            return
        if paused:
            return  # chair itself is now input-required; resumes via _resume

        await self._finish_with_critic(updater, task, shared_context_id, draft, sources)

    async def _resume(
        self, task: Task, context: RequestContext, event_queue: EventQueue
    ) -> None:
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        metadata = task.metadata or {}
        if metadata.get("stage") != "synthesis-input-required":
            logger.warning(
                "chair: resume with unknown pause stage %r on task %s",
                metadata.get("stage"),
                task.id,
            )
            await updater.failed(
                message=new_agent_text_message(
                    "Internal error: chair does not know what it was waiting for.",
                    task.context_id,
                    task.id,
                )
            )
            return

        topic = metadata.get("topic", "")
        sources = metadata.get("sources", [])
        synthesis_context_id = metadata.get("synthesis_context_id")
        synthesis_task_id = metadata.get("synthesis_task_id")
        guidance = context.get_user_input().strip()

        await updater.start_work()
        await self._relay(
            updater, task, f'[chair] forwarding your reply to synthesis: "{guidance}"'
        )

        follow_up = text_message(
            guidance, synthesis_context_id, task_id=synthesis_task_id
        )
        try:
            draft, paused = await self._stream_synthesis(
                updater, task, synthesis_context_id, follow_up, topic, sources
            )
        except SubAgentError as exc:
            await self._fail(updater, task, exc)
            return
        if paused:
            return  # synthesis asked again; stays input-required for another round

        await self._finish_with_critic(
            updater, task, synthesis_context_id, draft, sources
        )

    async def _finish_with_critic(
        self,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        draft: str,
        sources: list[dict],
    ) -> None:
        try:
            verdicts = await self._run_critic(updater, task, context_id, draft, sources)
        except SubAgentError as exc:
            await self._fail(updater, task, exc)
            return
        await self._finish(updater, task, draft, verdicts)

    # -- sub-agent calls -----------------------------------------------------

    async def _run_lit_search(
        self, updater: TaskUpdater, task: Task, context_id: str, topic: str
    ) -> list[dict]:
        message = text_message(topic, context_id)
        client = self._client_for("lit_search")
        sources: list[dict] = []
        try:
            async for client_event in client.send_message(message):
                if isinstance(client_event, Message):
                    continue
                _, update = client_event
                if update is None:
                    continue
                if isinstance(update, TaskStatusUpdateEvent):
                    if update.status.state == TaskState.working:
                        await self._relay(updater, task, "[lit_search] searching for sources…")
                    elif update.status.state == TaskState.failed:
                        raise SubAgentError(
                            "lit_search", message_text(update.status.message) or "task failed"
                        )
                    elif update.status.state == TaskState.completed:
                        await self._relay(
                            updater, task, f"[lit_search] found {len(sources)} source(s)."
                        )
                elif isinstance(update, TaskArtifactUpdateEvent):
                    for part in update.artifact.parts:
                        if isinstance(part.root, DataPart):
                            sources.append(part.root.data)
                            title = part.root.data.get("title", "?")
                            await self._relay(
                                updater,
                                task,
                                f"[lit_search] found source {len(sources)}: {title}",
                            )
        except _TRANSPORT_ERRORS as exc:
            raise SubAgentError("lit_search", str(exc)) from exc
        return sources

    async def _stream_synthesis(
        self,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        message: Message,
        topic: str,
        sources: list[dict],
    ) -> tuple[str | None, bool]:
        """Sends `message` to synthesis and streams its response.

        Returns (draft, paused). If synthesis pauses with input-required,
        chair pauses too, `draft` is None and `paused` is True.
        """
        client = self._client_for("synthesis")
        draft_parts: list[str] = []
        synthesis_task_id: str | None = None
        try:
            async for client_event in client.send_message(message):
                if isinstance(client_event, Message):
                    continue
                task_snapshot, update = client_event
                synthesis_task_id = task_snapshot.id
                if update is None:
                    continue
                if isinstance(update, TaskStatusUpdateEvent):
                    if update.status.state == TaskState.working:
                        await self._relay(updater, task, "[synthesis] drafting answer…")
                    elif update.status.state == TaskState.input_required:
                        question = message_text(update.status.message)
                        await self._pause_for_synthesis_input(
                            updater,
                            task,
                            context_id,
                            topic,
                            sources,
                            synthesis_task_id,
                            question,
                        )
                        return None, True
                    elif update.status.state == TaskState.failed:
                        raise SubAgentError(
                            "synthesis", message_text(update.status.message) or "task failed"
                        )
                elif isinstance(update, TaskArtifactUpdateEvent):
                    for part in update.artifact.parts:
                        if isinstance(part.root, TextPart):
                            draft_parts.append(part.root.text)
                            await self._relay(
                                updater, task, f"[synthesis] {part.root.text.strip()}"
                            )
        except _TRANSPORT_ERRORS as exc:
            raise SubAgentError("synthesis", str(exc)) from exc
        return "".join(draft_parts), False

    async def _run_critic(
        self,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        draft: str,
        sources: list[dict],
    ) -> list[dict]:
        message = data_message({"draft": draft, "sources": sources}, context_id)
        client = self._client_for("critic")
        verdicts: list[dict] = []
        try:
            async for client_event in client.send_message(message):
                if isinstance(client_event, Message):
                    continue
                _, update = client_event
                if update is None:
                    continue
                if isinstance(update, TaskStatusUpdateEvent):
                    if update.status.state == TaskState.working:
                        await self._relay(updater, task, "[critic] verifying claims…")
                    elif update.status.state == TaskState.failed:
                        raise SubAgentError(
                            "critic", message_text(update.status.message) or "task failed"
                        )
                elif isinstance(update, TaskArtifactUpdateEvent):
                    for part in update.artifact.parts:
                        if isinstance(part.root, DataPart):
                            verdicts = part.root.data.get("verdicts", [])
                            await self._relay(
                                updater,
                                task,
                                f"[critic] produced {len(verdicts)} verdict(s).",
                            )
        except _TRANSPORT_ERRORS as exc:
            raise SubAgentError("critic", str(exc)) from exc
        return verdicts

    # -- chair's own task updates --------------------------------------------

    async def _pause_for_synthesis_input(
        self,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        topic: str,
        sources: list[dict],
        synthesis_task_id: str | None,
        question: str,
    ) -> None:
        logger.info("chair: pausing for input — synthesis asked: %s", question)
        await updater.update_status(
            TaskState.input_required,
            message=new_agent_text_message(question, task.context_id, task.id),
            metadata={
                "stage": "synthesis-input-required",
                "topic": topic,
                "sources": sources,
                "synthesis_task_id": synthesis_task_id,
                "synthesis_context_id": context_id,
            },
        )

    async def _relay(self, updater: TaskUpdater, task: Task, text: str) -> None:
        await updater.update_status(
            TaskState.working,
            message=new_agent_text_message(text, task.context_id, task.id),
        )

    async def _fail(self, updater: TaskUpdater, task: Task, exc: SubAgentError) -> None:
        logger.error("chair: %s", exc)
        await updater.failed(
            message=new_agent_text_message(
                f"[{exc.agent_name}] {exc.detail}", task.context_id, task.id
            )
        )

    async def _finish(
        self, updater: TaskUpdater, task: Task, draft: str, verdicts: list[dict]
    ) -> None:
        await self._relay(updater, task, "[chair] finalizing answer and verdicts…")
        await updater.add_artifact(
            parts=[Part(root=DataPart(data={"answer": draft, "verdicts": verdicts}))],
            artifact_id=str(uuid4()),
            name="research-answer",
            last_chunk=True,
        )
        await updater.complete()
