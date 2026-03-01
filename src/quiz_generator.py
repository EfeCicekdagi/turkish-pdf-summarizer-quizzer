# src/quiz_generator.py
"""
Template-based Turkish quiz generator.

Does NOT use any LLM — works by extracting key sentences and terms
from the summary text and building questions programmatically.
This guarantees:
  - Answers exist in the source text (no hallucination)
  - Output is always in Turkish
  - No model loading overhead
"""
from __future__ import annotations

import random
import re
from typing import List, Tuple

from src.extractive import sentence_split, tfidf_scores, _tokenize


# ---------------------------------------------------------------------------
# Turkish stop-words (common function words that are bad answer keys)
# ---------------------------------------------------------------------------
_TR_STOPWORDS = {
    "bir", "ve", "bu", "ile", "için", "de", "da", "bu", "şu", "o",
    "olan", "olan", "gibi", "kadar", "daha", "en", "çok", "az",
    "biz", "siz", "onlar", "ben", "sen", "ise", "veya", "ya", "ki",
    "hem", "ne", "her", "hiç", "bazı", "tüm", "bütün", "birçok",
    "sadece", "yalnız", "ancak", "fakat", "ama", "lakin", "çünkü",
    "eğer", "ise", "olarak", "ayrıca", "öte", "yandan", "sonra",
    "önce", "göre", "karşı", "doğru", "içinde", "üzerinde",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(word: str) -> str:
    """Strip punctuation from a word."""
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE)


def _meaningful_words(sentence: str, min_len: int = 4) -> List[Tuple[int, str]]:
    """
    Return (index, word) pairs for content words (not stopwords, long enough).
    """
    results = []
    for i, raw in enumerate(sentence.split()):
        w = _clean(raw).lower()
        if w and len(w) >= min_len and w not in _TR_STOPWORDS:
            results.append((i, _clean(raw)))
    return results


def _mask_word(sentence: str, word_index: int) -> str:
    """Replace word at word_index with '___'."""
    parts = sentence.split()
    parts[word_index] = "___"
    return " ".join(parts)


def _pick_distractors(
    correct: str,
    pool: List[str],
    n: int = 3,
    rng: random.Random | None = None,
) -> List[str]:
    """
    Pick n distractor words from pool (different from correct).
    Falls back to generic placeholders if pool is small.
    """
    if rng is None:
        rng = random.Random(42)
    distractors = [w for w in pool if w.lower() != correct.lower() and len(w) >= 3]
    distractors = list(dict.fromkeys(distractors))  # deduplicate, preserve order
    rng.shuffle(distractors)
    chosen = distractors[:n]
    # Pad if not enough
    placeholders = ["—", "belirsiz", "yok"]
    while len(chosen) < n:
        chosen.append(placeholders[len(chosen) % len(placeholders)])
    return chosen


def _score_sentences(sentences: List[str]) -> List[Tuple[int, float]]:
    """Return (index, score) sorted by descending score."""
    scores = tfidf_scores(sentences)
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


def _top_words_from_text(text: str, n: int = 30) -> List[str]:
    """Return top-n most frequent content words from the whole text."""
    from collections import Counter
    tokens = [t for t in _tokenize(text) if t not in _TR_STOPWORDS and len(t) >= 4]
    return [w for w, _ in Counter(tokens).most_common(n)]


# ---------------------------------------------------------------------------
# Question builders
# ---------------------------------------------------------------------------

def _build_short_answer(sentence: str) -> Tuple[str, str] | None:
    """
    Cloze question: blank out the most important word.
    Returns (question_text, answer) or None if no usable word found.
    """
    mw = _meaningful_words(sentence)
    if not mw:
        return None
    # Pick the longest meaningful word (likely a key term)
    idx, word = max(mw, key=lambda x: len(x[1]))
    question = f"[Kısa Cevap] {_mask_word(sentence, idx)}"
    return question, word


def _build_multiple_choice(
    sentence: str,
    distractor_pool: List[str],
    rng: random.Random,
) -> Tuple[str, str] | None:
    """
    Multiple-choice question by blanking a key word and offering 4 choices.
    Returns (question_text_with_options, correct_letter) or None.
    """
    mw = _meaningful_words(sentence)
    if not mw:
        return None
    idx, word = max(mw, key=lambda x: len(x[1]))
    distractors = _pick_distractors(word, distractor_pool, n=3, rng=rng)
    options = [word] + distractors
    rng.shuffle(options)
    correct_letter = chr(ord("A") + options.index(word))
    opts_str = "\n   ".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
    question = f"[Çoktan Seçmeli] {_mask_word(sentence, idx)}\n   {opts_str}"
    return question, correct_letter


def _build_true_false(sentence: str) -> Tuple[str, str]:
    """
    True/False question using the sentence as-is (always True).
    """
    question = f"[Doğru/Yanlış] {sentence}"
    return question, "Doğru"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_quiz(text: str, n_questions: int = 5, seed: int = 0) -> str:
    """
    Generate a Turkish quiz from text without any LLM.

    Distribution (for n_questions=5):
      - 2 Kısa Cevap (cloze / fill-in-the-blank)
      - 2 Çoktan Seçmeli
      - 1 Doğru/Yanlış

    Parameters
    ----------
    text        : the summary or source text
    n_questions : total questions to generate (default 5)
    seed        : random seed for reproducibility

    Returns
    -------
    Formatted quiz string.
    """
    rng = random.Random(seed)
    sentences = sentence_split(text)

    if not sentences:
        return "Yeterli metin bulunamadı, quiz üretilemedi."

    scored = _score_sentences(sentences)
    distractor_pool = _top_words_from_text(text)

    # Plan: 2 short-answer, 2 mc, 1 tf (scale proportionally for other n_questions)
    n_sa = max(1, round(n_questions * 0.4))
    n_mc = max(1, round(n_questions * 0.4))
    n_tf = n_questions - n_sa - n_mc

    questions: List[Tuple[str, str]] = []
    used_indices: set = set()

    def next_sentence(exclude_used: bool = True) -> str | None:
        for idx, _ in scored:
            if not exclude_used or idx not in used_indices:
                used_indices.add(idx)
                return sentences[idx]
        # Fallback: allow reuse
        for idx, _ in scored:
            return sentences[idx]
        return None

    # Short answer
    for _ in range(n_sa):
        s = next_sentence()
        if s is None:
            break
        result = _build_short_answer(s)
        if result:
            questions.append(result)

    # Multiple choice
    for _ in range(n_mc):
        s = next_sentence()
        if s is None:
            break
        result = _build_multiple_choice(s, distractor_pool, rng)
        if result:
            questions.append(result)

    # True/False
    for _ in range(n_tf):
        s = next_sentence()
        if s is None:
            break
        questions.append(_build_true_false(s))

    if not questions:
        return "Quiz üretilemedi: metinde yeterli içerik bulunamadı."

    # Format output
    lines = []
    answer_key = []
    for i, (q_text, answer) in enumerate(questions, start=1):
        lines.append(f"{i}) {q_text}\n")
        answer_key.append(f"{i}) {answer}")

    lines.append("Cevap Anahtarı:")
    lines.extend(answer_key)
    return "\n".join(lines)
