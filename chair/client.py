"""A2A client-side helpers for the chair's calls to its three sub-agents.

Uses the SDK's ClientFactory/Client — the officially recommended client
API in the pinned a2a-sdk version (the older A2AClient is deprecated in
favor of it). Client.send_message already branches on streaming vs.
non-streaming based on the Agent Card's capabilities.streaming, so chair
doesn't need to hardcode which sub-agents stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.types import AgentCard


class SubAgentUnavailableError(RuntimeError):
    """Raised at startup when a downstream agent is unreachable or missing
    its expected skill — chair should fail loudly here, not mid-request."""


class SubAgentError(RuntimeError):
    """Raised when a call to a downstream agent fails during a live request."""

    def __init__(self, agent_name: str, detail: str) -> None:
        super().__init__(f"{agent_name}: {detail}")
        self.agent_name = agent_name
        self.detail = detail


@dataclass
class SubAgentSpec:
    name: str
    base_url: str
    skill_id: str


async def discover_agent_card(
    httpx_client: httpx.AsyncClient, spec: SubAgentSpec
) -> AgentCard:
    """Fetches and validates one downstream agent's Agent Card."""
    resolver = A2ACardResolver(httpx_client, spec.base_url)
    try:
        card = await resolver.get_agent_card()
    except Exception as exc:
        raise SubAgentUnavailableError(
            f"{spec.name}: could not fetch Agent Card from {spec.base_url} ({exc})"
        ) from exc

    skill_ids = {skill.id for skill in card.skills}
    if spec.skill_id not in skill_ids:
        raise SubAgentUnavailableError(
            f"{spec.name}: Agent Card at {spec.base_url} is missing expected "
            f"skill '{spec.skill_id}' (has: {sorted(skill_ids) or 'none'})"
        )
    return card


async def discover_all(
    httpx_client: httpx.AsyncClient, specs: list[SubAgentSpec]
) -> dict[str, AgentCard]:
    """Fetches and validates every sub-agent's Agent Card, failing on the first miss."""
    cards: dict[str, AgentCard] = {}
    for spec in specs:
        cards[spec.name] = await discover_agent_card(httpx_client, spec)
    return cards


def build_client_factory(httpx_client: httpx.AsyncClient) -> ClientFactory:
    return ClientFactory(ClientConfig(httpx_client=httpx_client))


def build_client(factory: ClientFactory, card: AgentCard) -> Client:
    """Builds a Client for one sub-agent from its already-validated Agent Card."""
    return factory.create(card)
