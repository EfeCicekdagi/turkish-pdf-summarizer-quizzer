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

    Parameters
    ----------
    pdf_path  : path to the temporary PDF file written by Streamlit
    max_pages : if provided, only the first N pages are processed

    Raises
    ------
    ValueError
        If the file cannot be opened (encrypted, corrupted, wrong format)
        or if no readable text could be extracted (image-only / scanned PDF).
    """
    # Try to open the file — fitz raises fitz.FileDataError for corrupted /
    # encrypted PDFs and plain RuntimeError for unsupported formats.
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(
            f"Could not open PDF file. It may be encrypted, corrupted, or "
            f"not a valid PDF. Original error: {exc}"
        ) from exc

    page_count = doc.page_count
    limit = page_count if max_pages is None else min(max_pages, page_count)

    parts = []
    for i in range(limit):
        page = doc.load_page(i)
        parts.append(page.get_text("text"))

    doc.close()

    text = clean_text("\n".join(parts))

    # Guard against image-only / scanned PDFs that contain no selectable text
    if not text:
        raise ValueError(
            "No readable text found in the PDF. "
            "The file may be a scanned image. Try an OCR tool first."
        )

    return PDFExtractResult(text=text, page_count=page_count, char_count=len(text))