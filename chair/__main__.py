"""Runs the chair A2A server on port 9000.

On startup, fetches and validates the Agent Cards of all three
sub-agents (lit_search, synthesis, critic) before accepting any human
request — per SPEC.md, chair fails loudly here (logs which agent/skill
is missing and exits) rather than discovering it mid-request.

Sub-agent base URLs can be overridden via LIT_SEARCH_URL / SYNTHESIS_URL
/ CRITIC_URL env vars — handy for deliberately testing the failure path
(point one at a wrong/closed port).
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard

from chair.agent_card import PORT, build_agent_card
from chair.agent_executor import ChairAgentExecutor
from chair.client import SubAgentSpec, SubAgentUnavailableError, discover_all
from common.logging_config import configure_logging

HOST = "0.0.0.0"

SUB_AGENT_SPECS = [
    SubAgentSpec(
        name="lit_search",
        base_url=os.environ.get("LIT_SEARCH_URL", "http://localhost:9001"),
        skill_id="find-sources",
    ),
    SubAgentSpec(
        name="synthesis",
        base_url=os.environ.get("SYNTHESIS_URL", "http://localhost:9002"),
        skill_id="draft-answer",
    ),
    SubAgentSpec(
        name="critic",
        base_url=os.environ.get("CRITIC_URL", "http://localhost:9003"),
        skill_id="verify-claims",
    ),
]

logger = logging.getLogger(__name__)


def validate_sub_agents() -> dict[str, AgentCard]:
    async def _validate() -> dict[str, AgentCard]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            return await discover_all(client, SUB_AGENT_SPECS)

    try:
        return asyncio.run(_validate())
    except SubAgentUnavailableError as exc:
        logger.error("chair: startup validation failed — %s", exc)
        raise SystemExit(1) from exc


def main() -> None:
    configure_logging()

    cards = validate_sub_agents()
    for name, card in cards.items():
        logger.info("chair: %s is reachable and capable (%s)", name, card.url)

    request_handler = DefaultRequestHandler(
        agent_executor=ChairAgentExecutor(cards),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
