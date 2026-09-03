"""Shared naive text-processing helpers.

Used by both the critic agent (claim/source keyword overlap) and the
synthesis agent (source contradiction detection). Deliberately crude,
keyword-level logic — not real NLP or an LLM call. See SPEC.md Step 3.
"""

from __future__ import annotations

import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "to",
    "and", "or", "for", "with", "that", "this", "it", "as", "by",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def split_sentences(text: str) -> list[str]:
    """Splits text into sentence-like fragments on '. '/'! '/'? ' boundaries."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def tokenize(text: str) -> set[str]:
    """Lowercases, strips punctuation, tokenizes on whitespace, drops stopwords."""
    tokens = _TOKEN_RE.findall(text.lower())
    return {t for t in tokens if t not in STOPWORDS}
