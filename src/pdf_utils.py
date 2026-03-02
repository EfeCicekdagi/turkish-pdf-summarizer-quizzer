# src/pdf_utils.py
"""
pdf_utils.py - PDF Text Extraction
====================================
Extracts plain text from PDF files using PyMuPDF (fitz).

Features:
    - max_pages parameter processes only the first N pages (useful for quick testing)
    - Extracted text is normalized to remove redundant whitespace
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import fitz  # PyMuPDF — install with: pip install pymupdf


@dataclass
class PDFExtractResult:
    """Container for the result of a PDF extraction."""
    text: str        # Full extracted and cleaned plain text
    page_count: int  # Total number of pages in the PDF
    char_count: int  # Character count of the extracted text


def clean_text(text: str) -> str:
    """
    Basic cleanup:
    - normalize whitespace
    - remove repeated empty lines
    """
    text = text.replace("\r", "\n")
    # collapse 3+ newlines to 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str, max_pages: Optional[int] = None) -> PDFExtractResult:
    """
    Extract text from a PDF file using PyMuPDF.
    max_pages: if provided, only extract the first N pages (useful for quick testing).
    """
    doc = fitz.open(pdf_path)
    page_count = doc.page_count

    limit = page_count if max_pages is None else min(max_pages, page_count)

    parts = []
    for i in range(limit):
        page = doc.load_page(i)
        parts.append(page.get_text("text"))

    doc.close()

    text = clean_text("\n".join(parts))
    return PDFExtractResult(text=text, page_count=page_count, char_count=len(text))