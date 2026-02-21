# src/prompts.py
from __future__ import annotations


def build_chunk_summarize_prompt(chunk_text: str) -> str:
    # FLAN-T5 için net, kısa talimatlar iyi çalışır
    return f"""
Aşağıdaki metni Türkçe olarak özetle.
- 5-8 madde kullan
- Gereksiz tekrar yapma
- Önemli kavramları koru

METİN:
{chunk_text}
""".strip()


def build_final_summarize_prompt(all_chunk_summaries: str) -> str:
    return f"""
Aşağıdaki özet parçalarını tek bir "yapılandırılmış final özet" haline getir.

ÇIKTI FORMATI:
Ana Fikir:
Temel Kavramlar: (virgülle)
Önemli Maddeler:
- ...
- ...
Kısa Özet (3-5 cümle):

ÖZET PARÇALARI:
{all_chunk_summaries}
""".strip()


def build_quiz_prompt(final_summary: str, n_questions: int = 5) -> str:
    # Formatı sabitlemek GitHub demosu için çok iyi durur
    return f"""
Aşağıdaki özete göre Türkçe bir mini quiz üret.

İSTEKLER:
- Toplam {n_questions} soru
- Soru dağılımı:
  1) 2 kısa cevap
  2) 2 çoktan seçmeli (A/B/C/D)
  3) 1 doğru/yanlış
- En sonda "Cevap Anahtarı:" yaz ve cevapları listele
- Sorular net ve özete bağlı olsun

ÖZET:
{final_summary}
""".strip()