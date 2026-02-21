# src/postprocess.py
from __future__ import annotations

import re


def normalize_output(text: str) -> str:
    text = text.replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()