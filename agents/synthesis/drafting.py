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

# --- crude content-hygiene filtering for snippets going into the draft ---
#
# Real search results (e.g. Tavily) come back as scraped page text, not
# clean sentences: markdown headers, nav/ad boilerplate, and bare
# numbered-list markers ("1.", "2.") show up as lines in the snippet. None
# of that is a claim or a sentence, so it shouldn't be woven into the
# templated draft. This is NOT real content extraction — just enough
# line-level filtering to keep obvious junk out before it hits the naive
# templating below, which was only ever designed for clean fake snippets.
# Messier pages will still let some noise through; that's an accepted v1
# limitation, not something to chase further right now.
_MIN_SNIPPET_LINE_WORDS = 4

_AD_KEYWORDS = (
    "no-code", "sign up", "sign in", "log in", "get started", "learn more",
    "subscribe", "pricing", "free trial", "book a demo", "contact us",
    "shop now", "buy now", "add to cart",
)

# A line with none of these is probably a brand/product name or nav label,
# not a sentence — even if it's long enough to pass the word-count check.
_COMMON_VERBS = {
    "is", "are", "was", "were", "be", "been", "has", "have", "had",
    "does", "do", "did", "shows", "show", "proposes", "introduces",
    "provides", "offers", "presents", "demonstrates", "uses", "makes",
    "notes", "states", "yields", "covers", "surveys", "increases",
    "decreases", "refutes", "confirms", "supports", "contradicts",
    "replacing", "replaces", "helps", "enables", "allows", "requires",
    "includes", "describes", "explains", "finds", "reports", "suggests",
    "argues", "claims", "means", "focuses", "aims", "seeks", "improves",
    "reduces", "achieves", "outperforms", "trains", "trained",
}


def _is_junk_line(line: str) -> bool:
    """True if `line` looks like markdown/nav/ad noise rather than prose."""
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    words = stripped.split()
    if len(words) < _MIN_SNIPPET_LINE_WORDS:
        return True
    lowered = stripped.lower()
    if any(keyword in lowered for keyword in _AD_KEYWORDS):
        return True
    if not any(re.sub(r"[^a-z]", "", w.lower()) in _COMMON_VERBS for w in words):
        return True
    return False


def clean_snippet(snippet: str) -> str:
    """Strips junk lines (headers, fragments, ad copy) out of a snippet
    before it's included in the draft. See `_is_junk_line` above."""
    lines = [line for line in snippet.splitlines() if not _is_junk_line(line)]
    return " ".join(line.strip() for line in lines)


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
            f'"{primary.get("title", "?")}", which notes: '
            f'{clean_snippet(primary.get("snippet", ""))}'
        ]
        for source in others:
            sentences.append(
                f'"{source.get("title", "?")}" adds further context: '
                f'{clean_snippet(source.get("snippet", ""))}'
            )
        return " ".join(sentences)

    if present_both:
        sentences = [f"Sources disagree on {topic}."]
        for source in sources:
            sentences.append(
                f'"{source.get("title", "?")}" states: '
                f'{clean_snippet(source.get("snippet", ""))}'
            )
        sentences.append(
            "Both perspectives are presented here without a single resolution."
        )
        return " ".join(sentences)

    sentences = [f"Regarding {topic}, several sources offer relevant perspectives."]
    for source in sources:
        sentences.append(
            f'"{source.get("title", "?")}" notes: '
            f'{clean_snippet(source.get("snippet", ""))}'
        )
    sentences.append(f"Taken together, these sources sketch an initial picture of {topic}.")
    return " ".join(sentences)
