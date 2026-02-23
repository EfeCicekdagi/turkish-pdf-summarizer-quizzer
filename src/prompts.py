# src/prompts.py
from __future__ import annotations


# NOTE:
# Fine-tuned summarization models (like many mT5 summarizers) often work best
# with RAW text or a very light "summarize:" prefix. Long instruction prompts
# can increase hallucinations. In our updated llm_pipeline.py we pass raw text.
# These functions are kept for experimentation / alternative models.


def build_chunk_summarize_prompt(chunk_text: str) -> str:
    """
    Use ONLY if your summarizer model is instruction-following.
    For many mT5 summarizers, prefer passing raw text directly.
    """
    # Minimal prefix style (safer than long instructions for T5-family)
    return f"summary: {chunk_text}".strip()


def build_final_summarize_prompt(all_chunk_summaries: str) -> str:
    """
    Use ONLY if your summarizer model follows instructions well.
    Otherwise, reduce summaries hierarchically with raw text (recommended).
    """
    # Keep it short; avoid giving the model room to invent structure content.
    return f"summary: {all_chunk_summaries}".strip()


def build_quiz_prompt(final_summary: str, n_questions: int = 5) -> str:
    """
    FLAN-T5 is instruction-tuned, so structured prompts help.
    Add anti-hallucination constraints: don't invent details not in the summary.
    """
    return f"""
Aşağıdaki ÖZET'e dayanarak Türkçe bir mini quiz üret.

KURALLAR (ÇOK ÖNEMLİ):
- Sadece ÖZET'te geçen bilgilere dayan. ÖZET'te olmayan kişi/kurum/olay/örnek uydurma.
- Soruların cevabı ÖZET'ten çıkarılabilir olmalı.
- Format dışına çıkma.

İSTEKLER:
- Toplam {n_questions} soru
- Dağılım:
  1) 2 kısa cevap
  2) 2 çoktan seçmeli (A/B/C/D)
  3) 1 doğru/yanlış
- En sonda "Cevap Anahtarı:" yaz ve cevapları 1..{n_questions} şeklinde listele.

ÇIKTI FORMATI (AYNEN UYGULA):
1) [Kısa Cevap] ...
2) [Kısa Cevap] ...
3) [Çoktan Seçmeli] ...
   A) ...
   B) ...
   C) ...
   D) ...
4) [Çoktan Seçmeli] ...
   A) ...
   B) ...
   C) ...
   D) ...
5) [Doğru/Yanlış] ...

Cevap Anahtarı:
1) ...
2) ...
3) ...
4) ...
5) ...

ÖZET:
{final_summary}
""".strip()