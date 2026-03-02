# src/extractive.py
"""
extractive.py - TF-IDF Extractive Summarizer
==============================================
Zero-hallucination summarization: always selects sentences from the
original text, never generates new ones.

Algorithm:
    1. Split text into individual sentences
    2. Score each sentence with TF-IDF
       (sentences that use rare/important words frequently score higher)
    3. Return the top-N sentences in their original reading order

Advantages:
    - No model loading required -> instant results
    - Output sentences come verbatim from the source text
    - Statistically driven -> language-agnostic (works for Turkish without stop-words)
"""
from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import List


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?\u2026])\s+"          # sentence-ending punctuation + whitespace
    r"|(?<=:)\s*\n"                  # colon + newline (list headers)
    r"|\n{2,}"                       # blank line (paragraph break)
)


def sentence_split(text: str) -> List[str]:
    """
    Split text into individual sentences.
    Filters out very short fragments (< 25 chars) that are likely headings or noise.
    """
    parts = _SENT_SPLIT_RE.split(text)
    sentences: List[str] = []
    for p in parts:
        p = p.strip()
        if p and len(p) >= 25:
            sentences.append(p)
    return sentences


# ---------------------------------------------------------------------------
# TF-IDF scoring
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """
    Lowercase + split on non-alpha characters.
    Simple but effective for Turkish (avoids nltk dependency).
    """
    return re.findall(r"[a-z\u00e7\u011f\u0131\u015f\u00f6\u00fc\u00e2\u00ee\u00fb]+", text.lower())


def tfidf_scores(sentences: List[str]) -> List[float]:
    """
    Score each sentence using TF-IDF.
    Higher score = sentence uses rare (important) words frequently.
    """
    n = len(sentences)
    if n == 0:
        return []

    tokenized = [_tokenize(s) for s in sentences]

    # Document frequency: how many sentences contain each word
    df: Counter = Counter()
    for words in tokenized:
        for w in set(words):
            df[w] += 1

    scores: List[float] = []
    for words in tokenized:
        if not words:
            scores.append(0.0)
            continue
        tf = Counter(words)
        # IDF: log(N / df) — rare words get higher weight
        score = sum(tf[w] * log((n + 1) / (df[w] + 1)) for w in words)
        # Normalize by sentence length to avoid bias toward long sentences
        score /= len(words)
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extractive_summary(text: str, n_sentences: int = 5) -> str:
    """
    Extract the top-N most important sentences from `text`,
    returned in their original order (no reordering = preserves coherence).

    Parameters
    ----------
    text        : input text (one chunk or full document)
    n_sentences : how many sentences to keep

    Returns
    -------
    Extracted sentences joined by a single space.
    If the text has fewer sentences than n_sentences, returns the full text.
    """
    sentences = sentence_split(text)

    if not sentences:
        return text.strip()

    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    scores = tfidf_scores(sentences)

    # Pick top-N indices, then sort to preserve original reading order
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted(idx for idx, _ in indexed[:n_sentences])

    return " ".join(sentences[i] for i in top_indices)
