"""Fake drafting + ambiguity-detection logic for the synthesis agent.

No real LLM call — a simple templated paragraph referencing the topic and
source titles, standing in for Step 5. Ambiguity detection reuses
common/text.py's tokenizer so it's consistent with critic's matching.
"""

from __future__ import annotations

import re

from common.text import tokenize

MIN_SOURCES = 2

POSITIVE_KEYWORDS = {"increases", "supports", "confirms"}
NEGATIVE_KEYWORDS = {"decreases", "refutes", "contradicts"}

# Opposite sentiment keywords alone are a weak signal — two sources on
# unrelated topics can each contain one by coincidence (e.g. "increases"
# describing solar panel efficiency, "decreases" describing recycling
# costs). Only treat it as a real disagreement if the sources also share
# enough topic vocabulary to plausibly be about the same thing. 0.15 was
# picked empirically: an unrelated pair scored ~0.08 overlap, a genuine
# same-topic disagreement scored ~0.20 — this sits between the two.
TOPIC_RELATEDNESS_THRESHOLD = 0.15

_SOURCE_INDEX_RE = re.compile(r"source\s*(\d+)", re.IGNORECASE)


def _source_tokens(source: dict) -> set[str]:
    return tokenize(f"{source.get('title', '')} {source.get('snippet', '')}")


def _sentiment(tokens: set[str]) -> str | None:
    if tokens & POSITIVE_KEYWORDS:
        return "positive"
    if tokens & NEGATIVE_KEYWORDS:
        return "negative"
    return None


def _topic_overlap_ratio(tokens_i: set[str], tokens_j: set[str]) -> float:
    """Containment-style overlap between two sources' non-sentiment tokens."""
    topic_i = tokens_i - POSITIVE_KEYWORDS - NEGATIVE_KEYWORDS
    topic_j = tokens_j - POSITIVE_KEYWORDS - NEGATIVE_KEYWORDS
    if not topic_i or not topic_j:
        return 0.0
    return len(topic_i & topic_j) / min(len(topic_i), len(topic_j))


def find_contradiction(sources: list[dict]) -> tuple[int, int] | None:
    """Returns the (i, j) indices of the first pair of sources with opposite
    sentiment keywords AND enough shared topic vocabulary to plausibly be
    about the same thing, or None if no such pair exists."""
    token_sets = [_source_tokens(s) for s in sources]
    sentiments = [_sentiment(t) for t in token_sets]
    for i in range(len(sentiments)):
        for j in range(i + 1, len(sentiments)):
            if sentiments[i] and sentiments[j] and sentiments[i] != sentiments[j]:
                ratio = _topic_overlap_ratio(token_sets[i], token_sets[j])
                if ratio >= TOPIC_RELATEDNESS_THRESHOLD:
                    return i, j
    return None


def ambiguity_prompt(sources: list[dict]) -> str | None:
    """Returns an input-required prompt if the sources are too thin or
    contradictory to draft from confidently, else None."""
    if len(sources) < MIN_SOURCES:
        return (
            f"Only {len(sources)} source(s) provided — proceed anyway, "
            "or wait for more?"
        )
    contradiction = find_contradiction(sources)
    if contradiction is not None:
        i, j = contradiction
        return (
            f"Sources {i} and {j} appear to disagree — should I "
            "prioritize one, or present both perspectives?"
        )
    return None


def parse_prioritized_index(guidance: str, sources: list[dict]) -> int | None:
    """Crudely extracts a source index the caller wants prioritized from a
    free-text reply, e.g. "prioritize source 0" or a source's own title."""
    match = _SOURCE_INDEX_RE.search(guidance)
    if match:
        index = int(match.group(1))
        if 0 <= index < len(sources):
            return index
        if 0 <= index - 1 < len(sources):
            return index - 1

    guidance_lower = guidance.lower()
    for index, source in enumerate(sources):
        title = source.get("title", "").strip().lower()
        if title and title in guidance_lower:
            return index
    return None


def build_draft(
    topic: str,
    sources: list[dict],
    prioritized_index: int | None = None,
    present_both: bool = False,
) -> str:
    """Builds a fake, templated multi-sentence draft answer."""
    if not sources:
        return f"There is not yet enough information to draft an answer about {topic}."

    if prioritized_index is not None and 0 <= prioritized_index < len(sources):
        primary = sources[prioritized_index]
        others = [s for i, s in enumerate(sources) if i != prioritized_index]
        sentences = [
            f'Regarding {topic}, the strongest evidence comes from '
            f'"{primary.get("title", "?")}", which notes: {primary.get("snippet", "")}'
        ]
        for source in others:
            sentences.append(
                f'"{source.get("title", "?")}" adds further context: '
                f'{source.get("snippet", "")}'
            )
        return " ".join(sentences)

    if present_both:
        sentences = [f"Sources disagree on {topic}."]
        for source in sources:
            sentences.append(
                f'"{source.get("title", "?")}" states: {source.get("snippet", "")}'
            )
        sentences.append(
            "Both perspectives are presented here without a single resolution."
        )
        return " ".join(sentences)

    sentences = [f"Regarding {topic}, several sources offer relevant perspectives."]
    for source in sources:
        sentences.append(f'"{source.get("title", "?")}" notes: {source.get("snippet", "")}')
    sentences.append(f"Taken together, these sources sketch an initial picture of {topic}.")
    return " ".join(sentences)
