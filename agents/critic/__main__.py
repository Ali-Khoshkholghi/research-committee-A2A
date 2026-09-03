"""Runs the critic A2A server on port 9003."""

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from agents.critic.agent_card import PORT, build_agent_card
from agents.critic.agent_executor import CriticAgentExecutor
from common.logging_config import configure_logging

HOST = "0.0.0.0"


def main() -> None:
    configure_logging()

    request_handler = DefaultRequestHandler(
        agent_executor=CriticAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=HOST, port=PORT)


if __name__ == "__main__":
    main()
