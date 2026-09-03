"""Agent Card for the critic agent."""

from a2a.types import AgentCapabilities, AgentCard, AgentSkill

PORT = 9003

VERIFY_CLAIMS_SKILL = AgentSkill(
    id="verify-claims",
    name="Verify claims",
    description=(
        "Checks each claim sentence in a drafted answer against the "
        "supplied sources and returns a supported/unsupported verdict "
        "with a note for each."
    ),
    tags=["research", "fact-checking"],
    examples=["Verify the claims in this draft against its sources"],
    input_modes=["application/json"],
    output_modes=["application/json"],
)


def build_agent_card(url: str = f"http://localhost:{PORT}/") -> AgentCard:
    """Builds the public Agent Card for the critic agent.

    capabilities.streaming is advertised as True for consistency with the
    other agents, even though this agent's own execution is non-streaming
    internally (one working status, one final artifact, then completed).
    """
    return AgentCard(
        name="critic",
        description=(
            "Claim-verification specialist. Given a drafted answer and its "
            "sources, checks each claim sentence for source support and "
            "returns a verdict list."
        ),
        url=url,
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[VERIFY_CLAIMS_SKILL],
    )
