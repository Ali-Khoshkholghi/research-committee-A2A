"""Naive claim-verification matching for the critic agent.

This is intentionally a crude keyword-overlap heuristic — not real
NLP or an LLM call — just enough to prove the DataPart-in/DataPart-out
shape works end-to-end (SPEC.md Step 3). Swap for a real fact-checking
approach later.
"""

from __future__ import annotations

from common.text import split_sentences, tokenize

SUPPORT_THRESHOLD = 0.4


def verify_claim(claim: str, sources: list[dict]) -> dict:
    """Scores a single claim sentence against every source, keeping the best match."""
    claim_tokens = tokenize(claim)
    if not sources or not claim_tokens:
        return {
            "claim": claim,
            "supported": False,
            "note": "Unable to verify — insufficient data",
        }

    best_source = sources[0]
    best_ratio = -1.0
    for source in sources:
        source_text = f"{source.get('title', '')} {source.get('snippet', '')}"
        source_tokens = tokenize(source_text)
        ratio = len(claim_tokens & source_tokens) / len(claim_tokens)
        if ratio > best_ratio:
            best_ratio = ratio
            best_source = source

    supported = best_ratio >= SUPPORT_THRESHOLD
    title = best_source.get("title", "?")
    if supported:
        note = f"Matches source: {title}"
    else:
        note = f"No source with sufficient overlap (best: {title}, ratio: {best_ratio:.2f})"
    return {"claim": claim, "supported": supported, "note": note}


def verify_claims(draft: str, sources: list[dict]) -> list[dict]:
    """Splits a drafted answer into claim sentences and verifies each one."""
    return [verify_claim(claim, sources) for claim in split_sentences(draft)]
