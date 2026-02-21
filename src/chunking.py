# src/chunking.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    id: int
    text: str
    char_len: int


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Paragraph split by blank lines.
    """
    # Normalize newlines
    text = text.replace("\r", "\n")
    # Split by 2+ newlines
    parts = re.split(r"\n\s*\n", text)
    # Clean each paragraph
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Build chunks close to chunk_size characters using paragraph boundaries.
    Adds overlap (last N chars from previous chunk) to preserve context.

    chunk_size: target max characters per chunk
    overlap: number of trailing characters from previous chunk to prepend to next
    """
    if chunk_size <= 200:
        raise ValueError("chunk_size too small; use >= 200.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    paragraphs = _split_into_paragraphs(text)

    chunks: List[Chunk] = []
    current = ""

    def _flush():
        nonlocal current
        if current.strip():
            chunks.append(Chunk(id=len(chunks), text=current.strip(), char_len=len(current.strip())))
        current = ""

    for p in paragraphs:
        # If adding this paragraph would exceed size, flush current chunk
        if current and (len(current) + 2 + len(p)) > chunk_size:
            _flush()

        # If a single paragraph is too large, hard-split it
        if len(p) > chunk_size:
            # flush whatever we have
            _flush()
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                part = p[start:end].strip()
                if part:
                    chunks.append(Chunk(id=len(chunks), text=part, char_len=len(part)))
                start = end
            continue

        # Add paragraph
        current = (current + "\n\n" + p) if current else p

    _flush()

    # Apply overlap
    if overlap > 0 and len(chunks) > 1:
        new_chunks: List[Chunk] = []
        prev_text = ""
        for c in chunks:
            if prev_text:
                prefix = prev_text[-overlap:]
                merged = (prefix + "\n" + c.text).strip()
                new_chunks.append(Chunk(id=c.id, text=merged, char_len=len(merged)))
            else:
                new_chunks.append(c)
            prev_text = c.text
        chunks = new_chunks

    return chunks