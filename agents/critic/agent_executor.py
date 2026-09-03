"""AgentExecutor for the critic agent.

Step 3 of the SPEC. Non-streaming internally: one `working` status, one
final artifact, then `completed`. No input-required branch — critic
always has enough to produce a (possibly all-unsupported) verdict list.
"""

from __future__ import annotations

import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, UnsupportedOperationError
from a2a.utils import new_task
from a2a.utils.errors import ServerError

from agents.critic.matching import verify_claims
from common.parts import get_data_part

logger = logging.getLogger(__name__)

# TODO(v1): no auth/security hardening yet — see SPEC.md.


class CriticAgentExecutor(AgentExecutor):
    """Verifies each claim sentence in a draft answer against its sources."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()

        payload = get_data_part(context.message) or {}
        draft = payload.get("draft", "")
        sources = payload.get("sources", [])

        verdicts = verify_claims(draft, sources)
        logger.info("critic: verified %d claim(s)", len(verdicts))

        await updater.add_artifact(
            parts=[Part(root=DataPart(data={"verdicts": verdicts}))],
            artifact_id=str(uuid4()),
            name="verdicts",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise ServerError(error=UnsupportedOperationError())
