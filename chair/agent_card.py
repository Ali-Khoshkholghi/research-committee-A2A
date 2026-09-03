"""Agent Card for the chair agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

PORT = 9000

RESEARCH_QUESTION_SKILL = AgentSkill(
    id="research-question",
    name="Research question",
    description=(
        "Answers a research question by delegating to a committee of "
        "specialist agents (lit_search, synthesis, critic) over A2A, "
        "relaying their progress, and returning a synthesized, "
        "fact-checked answer."
    ),
    tags=["research", "orchestration"],
    examples=["What does the evidence say about sleep and memory consolidation?"],
    input_modes=["text/plain"],
    output_modes=["application/json"],
)


def build_agent_card(url: str = f"http://localhost:{PORT}/") -> AgentCard:
    """Builds the public Agent Card for the chair agent."""
    return AgentCard(
        name="chair",
        description=(
            "Research committee chair. Takes a research question from a "
            "human, delegates to lit_search, synthesis, and critic over "
            "A2A, relays their progress upward, and returns a "
            "synthesized, checked answer."
        ),
        url=url,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[RESEARCH_QUESTION_SKILL],
    )
