# src/i18n.py
"""
UI string translations.
Usage:
    from src.i18n import T
    t = T["tr"]  # or T["en"]
    st.title(t["title"])
"""
from __future__ import annotations

T: dict[str, dict[str, str]] = {
    "tr": {
        # page
        "page_title":        "Türkçe PDF Özetleyici + Quiz",
        "title":             "📄 Türkçe PDF Özetleyici + Quiz Üretici",
        "caption":           "PDF yükle → chunk'lara böl → özetle → sorular üret",
        # sidebar
        "settings":          "Ayarlar",
        "summarizer_model":  "Özet Modeli (TR)",
        "quiz_model":        "Quiz Modeli",
        "max_pages":         "Maks. sayfa sayısı (test için)",
        "chunk_size":        "Chunk boyutu (karakter)",
        "overlap":           "Overlap (kelime sayısı)",
        "first_n_chunks":    "Hızlı demo: ilk N chunk",
        "summary_method":    "Özet Yöntemi",
        "extractive_label":  "Çıkarımsal — güvenilir, halüsinasyon yok",
        "abstractive_label": "Üretimsel (mT5) — akıcı ama yanıltıcı olabilir",
        "extractive_help":   "Çıkarımsal: orijinal metinden cümleler seçilir.\nÜretimsel: mT5 yeni cümle üretir.",
        "sentences_per_chunk": "Chunk başına cümle sayısı",
        "sentences_help":    "Sadece Çıkarımsal modda geçerlidir.",
        "n_questions":       "Soru sayısı",
        "tip":               "İpucu: Büyük PDF'lerde `max sayfa` ve `ilk N chunk` değerini küçük tut.",
        # left column
        "step1":             "1) PDF yükle veya metin yapıştır",
        "upload_label":      "PDF yükle",
        "paste_label":       "...veya buraya metin yapıştır",
        "paste_placeholder": "PDF yoksa buraya metin yapıştırabilirsin.",
        "show_text":         "Çıkarılan metni göster",
        "extracted_ok":      "Metin çıkarıldı ✅  Sayfa: {pages} | Karakter: {chars}",
        "pasted_ok":         "Yapıştırılan metin alındı ✅  Karakter: {chars}",
        "extracted_preview": "Çıkarılan metin (önizleme)",
        "step2":             "2) Chunking",
        "chunk_count":       "Chunk sayısı: **{n}**",
        "first_chunk":       "İlk chunk önizlemesi",
        # right column
        "step3":             "3) Özet + Quiz üret",
        "btn_summarize":     "🧠 Özetle",
        "btn_quiz":          "📝 Quiz Üret",
        "spinner_summarize": "Chunk'lar özetleniyor...",
        "spinner_quiz":      "Quiz üretiliyor...",
        "progress_chunk":    "Chunk özetleniyor: {done} / {total}",
        "summary_ready":     "Özet hazır ✅",
        "quiz_ready":        "Quiz hazır ✅",
        "final_summary":     "Final Özet",
        "download_summary":  "⬇️ Özeti indir (txt)",
        "quiz_section":      "Quiz",
        "download_quiz":     "⬇️ Quiz indir (txt)",
        "chunk_debug":       "Chunk özetlerini göster (debug)",
        "chunk_debug_item":  "**Chunk Özet {i}**",
        "pdf_error":         "PDF okunamadı",
        "no_chunks_warning": "Chunk bulunamadı. Lütfen bir PDF yükleyin veya metin yapıştırın.",
        # language selector
        "lang_label":        "Dil / Language",
    },
    "en": {
        # page
        "page_title":        "Turkish PDF Summarizer + Quiz",
        "title":             "📄 Turkish PDF Summarizer + Quiz Generator",
        "caption":           "Upload PDF → split into chunks → summarize → generate questions",
        # sidebar
        "settings":          "Settings",
        "summarizer_model":  "Summarizer Model (TR)",
        "quiz_model":        "Quiz Model",
        "max_pages":         "Max pages to extract (for testing)",
        "chunk_size":        "Chunk size (characters)",
        "overlap":           "Overlap (word count)",
        "first_n_chunks":    "Quick demo: first N chunks",
        "summary_method":    "Summary Method",
        "extractive_label":  "Extractive — reliable, no hallucination",
        "abstractive_label": "Abstractive (mT5) — fluent but may hallucinate",
        "extractive_help":   "Extractive: selects sentences from original text.\nAbstractive: mT5 generates new sentences.",
        "sentences_per_chunk": "Sentences per chunk",
        "sentences_help":    "Only used in Extractive mode.",
        "n_questions":       "Number of questions",
        "tip":               "Tip: For large PDFs, keep `max pages` and `first N chunks` small.",
        # left column
        "step1":             "1) Upload PDF or paste text",
        "upload_label":      "Upload PDF",
        "paste_label":       "...or paste text here",
        "paste_placeholder": "If you don't have a PDF, you can paste text here.",
        "show_text":         "Show extracted text",
        "extracted_ok":      "Text extracted ✅  Pages: {pages} | Characters: {chars}",
        "pasted_ok":         "Pasted text received ✅  Characters: {chars}",
        "extracted_preview": "Extracted text (preview)",
        "step2":             "2) Chunking",
        "chunk_count":       "Number of chunks: **{n}**",
        "first_chunk":       "First chunk preview",
        # right column
        "step3":             "3) Summarize + Generate Quiz",
        "btn_summarize":     "🧠 Summarize",
        "btn_quiz":          "📝 Generate Quiz",
        "spinner_summarize": "Summarizing chunks...",
        "spinner_quiz":      "Generating quiz...",
        "progress_chunk":    "Summarizing chunk: {done} / {total}",
        "summary_ready":     "Summary ready ✅",
        "quiz_ready":        "Quiz ready ✅",
        "final_summary":     "Final Summary",
        "download_summary":  "⬇️ Download summary (txt)",
        "quiz_section":      "Quiz",
        "download_quiz":     "⬇️ Download quiz (txt)",
        "chunk_debug":       "Show chunk summaries (debug)",
        "chunk_debug_item":  "**Chunk Summary {i}**",
        "pdf_error":         "Could not read PDF",
        "no_chunks_warning": "No text chunks to summarize. Please upload a PDF or paste some text.",
        # language selector
        "lang_label":        "Dil / Language",
    },
}
