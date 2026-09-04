"""Runs the synthesis A2A server on port 9002.

Selects the drafting provider (fake templater, or real via Gemini) at
startup and fails loudly — same pattern as chair's Agent Card
validation and lit_search's search provider — if it's misconfigured,
rather than discovering it on the first draft request.
"""

import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv

from agents.synthesis.agent_card import PORT, build_agent_card
from agents.synthesis.agent_executor import SynthesisAgentExecutor
from agents.synthesis.llm_provider import SynthesisProviderConfigError, build_provider
from common.logging_config import configure_logging

load_dotenv()

HOST = "0.0.0.0"

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    try:
        provider = build_provider()
    except SynthesisProviderConfigError as exc:
        logger.error("synthesis: startup configuration failed — %s", exc)
        raise SystemExit(1) from exc

    logger.info("synthesis: using synthesis provider %s", type(provider).__name__)

    request_handler = DefaultRequestHandler(
        agent_executor=SynthesisAgentExecutor(provider),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
