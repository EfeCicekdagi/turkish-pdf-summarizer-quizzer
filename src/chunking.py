# src/chunking.py
"""
chunking.py - Text Chunking
============================
Splits large texts into overlapping segments so they fit within
the limited context windows (token limits) of LLMs.

Approach:
    1. Split text at paragraph boundaries
    2. Start a new chunk when the target chunk_size is reached
    3. Prepend the last overlap_words words of the previous chunk
       to the next one to preserve context continuity
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """Represents a single text segment."""
    id: int       # Zero-based chunk index
    text: str     # Chunk content (includes leading overlap from previous chunk)
    char_len: int # Character length of this chunk


_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def _normalize_text(text: str) -> str:
    """
    Normalize newlines + collapse excessive spaces.
    Keeps newlines because we use them for paragraphing.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse horizontal whitespace
    text = _WS_RE.sub(" ", text)
    # normalize too many blank lines
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Prefer paragraph split by blank lines.
    Fallback: if PDF extraction produced almost no blank lines,
    group lines into pseudo-paragraphs.
    """
    text = _normalize_text(text)

    # Primary: split by blank lines
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) >= 2:
        return parts

    # Fallback: PDF often has line-wrapped text with single newlines
    # Heuristic: join lines into blocks; keep a break when a line ends with
    # sentence punctuation OR the next line looks like a new section/bullet.
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []

    paragraphs: List[str] = []
    buf: List[str] = []

    def looks_like_new_block(line: str) -> bool:
        return bool(re.match(r"^(\d+[\.\)]|[-•*])\s+", line)) or line.isupper()

    for i, ln in enumerate(lines):
        buf.append(ln)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        end_of_sentence = ln.endswith((".", "!", "?", "…", ":", ";"))
        next_is_new = bool(nxt) and looks_like_new_block(nxt)

        if end_of_sentence or next_is_new:
            paragraphs.append(" ".join(buf).strip())
            buf = []

    if buf:
        paragraphs.append(" ".join(buf).strip())

    return [p for p in paragraphs if p]


def _smart_split_long_text(text: str, max_chars: int) -> List[str]:
    """
    Split a too-long paragraph into pieces without cutting words if possible.
    Tries to split near max_chars at a whitespace boundary.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    out: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        target_end = min(start + max_chars, n)
        if target_end == n:
            out.append(text[start:].strip())
            break

        # Try to find a good split point (whitespace) near the end
        split_at = text.rfind(" ", start, target_end)
        # If no whitespace found (very rare), hard cut
        if split_at == -1 or split_at <= start + 50:
            split_at = target_end

        out.append(text[start:split_at].strip())
        start = split_at

    return [p for p in out if p]


def _tail_words(text: str, word_count: int) -> str:
    """
    Return last N words from text (safer than last N chars for PDFs).
    """
    words = text.split()
    if len(words) <= word_count:
        return text.strip()
    return " ".join(words[-word_count:]).strip()


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap_words: int = 40,
) -> List[Chunk]:
    """
    Build chunks close to chunk_size characters using paragraph boundaries.
    Adds overlap (last N words from previous chunk) to preserve context.

    chunk_size: target max characters per chunk
    overlap_words: number of trailing words from previous chunk to prepend to next
    """
    if chunk_size <= 200:
        raise ValueError("chunk_size too small; use >= 200.")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0.")

    paragraphs = _split_into_paragraphs(text)

    chunks: List[Chunk] = []
    current = ""

    def flush():
        nonlocal current
        t = current.strip()
        if t:
            chunks.append(Chunk(id=len(chunks), text=t, char_len=len(t)))
        current = ""

    for p in paragraphs:
        # If a paragraph is too large, split it smartly first
        if len(p) > chunk_size:
            flush()
            for part in _smart_split_long_text(p, chunk_size):
                if part:
                    chunks.append(Chunk(id=len(chunks), text=part, char_len=len(part)))
            continue

        # If adding paragraph exceeds chunk_size, flush
        if current and (len(current) + 2 + len(p)) > chunk_size:
            flush()

        current = (current + "\n\n" + p) if current else p

    flush()

    # Apply overlap (word-based)
    if overlap_words > 0 and len(chunks) > 1:
        new_chunks: List[Chunk] = []
        prev_text = ""

        for c in chunks:
            if prev_text:
                prefix = _tail_words(prev_text, overlap_words)
                merged = (prefix + "\n" + c.text).strip()
                new_chunks.append(Chunk(id=c.id, text=merged, char_len=len(merged)))
            else:
                new_chunks.append(c)
            prev_text = c.text

        chunks = new_chunks

    return chunks