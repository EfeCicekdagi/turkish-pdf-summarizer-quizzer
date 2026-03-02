# src/postprocess.py
"""
postprocess.py - LLM Output Cleaner
=====================================
Normalizes raw text returned by a generative model before displaying it
in the Streamlit UI:
    - Unifies line endings  (\\r\\n -> \\n)
    - Collapses 3+ consecutive blank lines to 2
    - Reduces multiple spaces/tabs to a single space
"""
from __future__ import annotations

import re


def normalize_output(text: str) -> str:
    """Clean and normalize raw model output for display."""
    # Unify Windows-style line endings to Unix
    text = text.replace("\r", "\n").strip()
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple whitespace characters to a single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()