"""Runs the lit_search A2A server on port 9001.

Selects the search provider (fake, or real via Tavily) at startup and
fails loudly — same pattern as chair's Agent Card validation — if it's
misconfigured, rather than discovering it on the first search request.
"""

import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv

from agents.lit_search.agent_card import PORT, build_agent_card
from agents.lit_search.agent_executor import LitSearchAgentExecutor
from agents.lit_search.search_provider import SearchProviderConfigError, build_provider
from common.logging_config import configure_logging

load_dotenv()

HOST = "0.0.0.0"

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logging()

    try:
        provider = build_provider()
    except SearchProviderConfigError as exc:
        logger.error("lit_search: startup configuration failed — %s", exc)
        raise SystemExit(1) from exc

    logger.info("lit_search: using search provider %s", type(provider).__name__)

    request_handler = DefaultRequestHandler(
        agent_executor=LitSearchAgentExecutor(provider),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
