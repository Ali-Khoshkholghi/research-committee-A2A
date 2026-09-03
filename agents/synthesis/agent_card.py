"""Agent Card for the synthesis agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

PORT = 9002

DRAFT_ANSWER_SKILL = AgentSkill(
    id="draft-answer",
    name="Draft answer",
    description=(
        "Drafts a paragraph answering a research topic from a list of "
        "sources, streaming it out as it's written. Asks for guidance "
        "first if the sources are too thin or contradictory."
    ),
    tags=["research", "synthesis"],
    examples=["Draft an answer about sleep and memory from these sources"],
    input_modes=["application/json"],
    output_modes=["text/plain"],
)


def build_agent_card(url: str = f"http://localhost:{PORT}/") -> AgentCard:
    """Builds the public Agent Card for the synthesis agent."""
    return AgentCard(
        name="synthesis",
        description=(
            "Synthesis specialist. Given a topic and a list of sources, "
            "drafts an answer paragraph, streaming it out as text chunks. "
            "May pause with input-required if the sources are too thin or "
            "contradictory."
        ),
        url=url,
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[DRAFT_ANSWER_SKILL],
    )
