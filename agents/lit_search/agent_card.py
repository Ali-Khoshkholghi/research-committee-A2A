"""Agent Card for the lit_search agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

PORT = 9001

FIND_SOURCES_SKILL = AgentSkill(
    id="find-sources",
    name="Find sources",
    description=(
        "Searches for sources relevant to a research topic and returns "
        "them one at a time as they are found."
    ),
    tags=["research", "literature-search"],
    examples=["Find sources on the impact of sleep on memory consolidation"],
    input_modes=["text/plain"],
    output_modes=["application/json"],
)


def build_agent_card(url: str = f"http://localhost:{PORT}/") -> AgentCard:
    """Builds the public Agent Card for the lit_search agent."""
    return AgentCard(
        name="lit_search",
        description=(
            "Literature-search specialist. Given a research topic, streams "
            "back candidate sources (title, url, snippet) one at a time."
        ),
        url=url,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[FIND_SOURCES_SKILL],
    )
