"""Search provider abstraction for lit_search.

Step 5 of the SPEC: swap the fake source generator for real web search
without lit_search's executor (streaming/task logic) needing to know
which provider is behind it — it only ever calls `search()` and gets
back a list of {title, url, snippet} dicts, or a SearchProviderError it
turns into a failed task.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

DEFAULT_MAX_RESULTS = 4


class SearchProviderConfigError(RuntimeError):
    """Raised at startup when the selected provider is misconfigured
    (e.g. LIT_SEARCH_PROVIDER=tavily but no TAVILY_API_KEY) — lit_search
    should fail loudly here, the same pattern as chair's Agent Card
    validation, rather than discovering it on the first search."""


class SearchProviderError(RuntimeError):
    """Raised when a live search call fails: no results, a rate limit or
    API error, or a network failure. Caught by the executor and mapped
    to a failed task state with a clear message."""


class SearchProvider(ABC):
    """A source of {title, url, snippet} results for a query."""

    @abstractmethod
    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Returns up to `max_results` {title, url, snippet} dicts.

        Raises SearchProviderError if the search cannot produce results.
        """


class FakeSearchProvider(SearchProvider):
    """Deterministic fake results — no API key required.

    The default provider, and useful for running debug_client.py without
    burning real search API quota.
    """

    # Intentionally the same fixture used since Step 1.
    _FAKE_SOURCES: list[dict[str, Any]] = [
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "snippet": (
                "Introduces the Transformer architecture, replacing recurrence "
                "and convolutions with self-attention."
            ),
        },
        {
            "title": "Deep Residual Learning for Image Recognition",
            "url": "https://arxiv.org/abs/1512.03385",
            "snippet": (
                "Proposes residual connections that make very deep "
                "convolutional networks trainable."
            ),
        },
        {
            "title": "Language Models are Few-Shot Learners",
            "url": "https://arxiv.org/abs/2005.14165",
            "snippet": (
                "Shows that scaling autoregressive language models yields "
                "strong few-shot task performance without fine-tuning."
            ),
        },
        {
            "title": "A Survey of Large Language Models",
            "url": "https://arxiv.org/abs/2303.18223",
            "snippet": (
                "Surveys the recent development of large language models, "
                "covering pre-training, adaptation, and evaluation."
            ),
        },
    ]

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        return self._FAKE_SOURCES[:max_results]


class TavilySearchProvider(SearchProvider):
    """Real web search via the Tavily Search API (https://tavily.com)."""

    DEFAULT_API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str,
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds

    async def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                response = await client.post(self._api_url, json=payload)
        except httpx.TimeoutException as exc:
            raise SearchProviderError(
                f"Tavily search timed out after {self._timeout_seconds}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchProviderError(f"network error calling Tavily: {exc}") from exc

        if response.status_code == 401:
            raise SearchProviderError(
                "Tavily rejected the API key (HTTP 401) — check TAVILY_API_KEY"
            )
        if response.status_code == 429:
            raise SearchProviderError("Tavily API rate limit exceeded (HTTP 429)")
        if response.status_code >= 400:
            raise SearchProviderError(
                f"Tavily API error (HTTP {response.status_code}): {response.text[:200]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise SearchProviderError(f"Tavily returned invalid JSON: {exc}") from exc

        results = data.get("results") or []
        if not results:
            raise SearchProviderError(f"no results found for query: {query!r}")

        return [
            {
                "title": item.get("title") or "(untitled)",
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in results[:max_results]
        ]


def build_provider() -> SearchProvider:
    """Selects the search provider from LIT_SEARCH_PROVIDER (default: fake).

    Fails loudly (SearchProviderConfigError) if 'tavily' is selected but
    TAVILY_API_KEY isn't set, so misconfiguration is caught at startup —
    the caller should treat this the same as chair's Agent Card
    validation failure: log clearly and exit, don't start serving.
    """
    provider_name = os.environ.get("LIT_SEARCH_PROVIDER", "fake").strip().lower()

    if provider_name == "fake":
        return FakeSearchProvider()

    if provider_name == "tavily":
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise SearchProviderConfigError(
                "LIT_SEARCH_PROVIDER=tavily but TAVILY_API_KEY is not set. "
                "Set TAVILY_API_KEY, or set LIT_SEARCH_PROVIDER=fake (or "
                "leave it unset) to use the built-in fake generator instead."
            )
        # Optional override, mainly for pointing at a local stub in tests —
        # defaults to the real Tavily endpoint.
        api_url = os.environ.get("TAVILY_API_URL", TavilySearchProvider.DEFAULT_API_URL)
        return TavilySearchProvider(api_key=api_key, api_url=api_url)

    raise SearchProviderConfigError(
        f"Unknown LIT_SEARCH_PROVIDER={provider_name!r} (expected 'fake' or 'tavily')"
    )
