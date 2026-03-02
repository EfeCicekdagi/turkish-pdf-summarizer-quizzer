"""
app.py - Turkish PDF Summarizer + Quiz Generator
=================================================
Main Streamlit application file.

Flow:
    1. Upload a PDF (or paste text directly)
    2. Split text into overlapping chunks
    3. Summarize each chunk (Extractive or Abstractive)
    4. Generate quiz questions from the final summary

Dependencies:
    - src/pdf_utils.py    -> PDF text extraction (PyMuPDF)
    - src/chunking.py     -> Split text into overlapping chunks
    - src/llm_pipeline.py -> mT5 summarization + quiz generation
    - src/i18n.py         -> TR / EN UI translations
"""

import os
import tempfile

import torch
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.chunking import chunk_text
from src.llm_pipeline import LLMService
from src.i18n import T

# ── LANGUAGE SELECTION ────────────────────────────────────────────────────────
# Must run before set_page_config so the browser tab title is in the right language.
# The selection is persisted across reruns via session_state.
# ─────────────────────────────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Language toggle at the very top of the sidebar
_lang_choice = st.sidebar.radio(
    "Dil / Language",
    options=["English", "Türkçe"],
    horizontal=True,
    index=0 if st.session_state.lang == "en" else 1,
    key="_lang_radio",
)
st.session_state.lang = "en" if _lang_choice == "English" else "tr"

# Short alias — access translations via t["key"]
t = T[st.session_state.lang]

# ── PAGE CONFIGURATION ────────────────────────────────────────────────────────
# set_page_config MUST be the very first Streamlit call (except for the sidebar
# radio above which is needed to determine the page title language).
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=t["page_title"], layout="wide")

# Show CUDA availability in the sidebar for debugging / informational purposes
st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("GPU:", torch.cuda.get_device_name(0))

st.title(t["title"])
st.caption(t["caption"])

# ── SIDEBAR SETTINGS ─────────────────────────────────────────────────────────
# All model and processing parameters are configured here by the user.
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(t["settings"])

    # Turkish summarization model selection:
    # mukayese/mt5-base  → better quality, slower
    # ozcangundes/mt5-small → faster, lower quality
    summarizer_model = st.selectbox(
        t["summarizer_model"],
        options=[
            "mukayese/mt5-base-turkish-summarization",
            "ozcangundes/mt5-small-turkish-summarization",
        ],
        index=0,
    )

    # Quiz model selection — currently unused because quiz generation is
    # template-based (FLAN-T5 does not understand Turkish well enough).
    # Kept here for future LLM-based quiz integration.
    quiz_model = st.selectbox(
        t["quiz_model"],
        options=[
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=0,
    )

    # Limit page extraction to speed up testing on large documents
    max_pages = st.number_input(t["max_pages"], min_value=1, max_value=500, value=10, step=1)

    # chunk_size: target character length per chunk
    # overlap: number of trailing words from the previous chunk prepended to the next
    chunk_size = st.slider(t["chunk_size"], min_value=600, max_value=2000, value=1200, step=100)
    overlap = st.slider(t["overlap"], min_value=0, max_value=80, value=30, step=5)

    # Process only the first N chunks for a quick demo on large documents
    use_first_n_chunks = st.number_input(t["first_n_chunks"], min_value=1, max_value=50, value=6, step=1)

    st.divider()
    st.markdown(f"**{t['summary_method']}**")

    # Extractive: picks original sentences via TF-IDF → no hallucination, fast
    # Abstractive: generates new sentences with mT5 → fluent but slower and may hallucinate
    summary_mode_label = st.radio(
        label=t["summary_method"],
        options=[t["extractive_label"], t["abstractive_label"]],
        index=0,
        label_visibility="collapsed",
        help=t["extractive_help"],
    )
    summary_mode = "extractive" if summary_mode_label == t["extractive_label"] else "abstractive"

    # Number of sentences to extract per chunk (only active in Extractive mode)
    n_sentences = st.slider(
        t["sentences_per_chunk"],
        min_value=2, max_value=10, value=5, step=1,
        disabled=(summary_mode != "extractive"),
        help=t["sentences_help"],
    )

    # Total number of quiz questions to generate
    n_questions = st.slider(
        t["n_questions"],
        min_value=3, max_value=15, value=5, step=1,
    )

    st.divider()
    st.write(t["tip"])


# ── MODEL LOADING ─────────────────────────────────────────────────────────────
# @st.cache_resource loads the model weights once and reuses them across reruns.
# The cache is automatically invalidated when the model name changes.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_llm(summarizer_model: str, quiz_model: str) -> LLMService:
    """Load and cache both models. Re-runs when either model name changes."""
    return LLMService(summarizer_model=summarizer_model, quiz_model=quiz_model)


# ── MAIN LAYOUT: TWO COLUMNS ──────────────────────────────────────────────────
# Left column  → Input (PDF / text) + Chunking preview
# Right column → Summarization + Quiz generation + Downloads
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

# ── LEFT COLUMN: INPUT AND CHUNK VIEW ────────────────────────────────────────
with col_left:
    st.subheader(t["step1"])

    uploaded = st.file_uploader(t["upload_label"], type=["pdf"])
    pasted_text = st.text_area(t["paste_label"], height=220, placeholder=t["paste_placeholder"])

    show_extracted_text = st.checkbox(t["show_text"], value=False)

    if uploaded is not None:
        # Streamlit's UploadedFile is not a real file path, so we write it to
        # a temporary file first, then pass the path to PyMuPDF.
        # The finally block ensures the temp file is always deleted, even on error.
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name  # store path so PyMuPDF can open it

            extract_res = extract_text_from_pdf(tmp_path, max_pages=int(max_pages))
            raw_text = extract_res.text

            st.success(t["extracted_ok"].format(pages=extract_res.page_count, chars=extract_res.char_count))

            # Optionally show a preview of the first 8000 characters
            if show_extracted_text:
                st.text_area(t["extracted_preview"], raw_text[:8000], height=250)

        except Exception as exc:
            # Handles encrypted, corrupted or unsupported PDF formats gracefully
            st.error(f"{t.get('pdf_error', 'Could not read PDF')}: {exc}")
            raw_text = ""

        finally:
            # Always clean up the temp file regardless of success or error
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        # Fall back to pasted text when no PDF is uploaded
        raw_text = pasted_text.strip()
        if raw_text:
            st.info(t["pasted_ok"].format(chars=len(raw_text)))

    st.divider()
    st.subheader(t["step2"])

    if raw_text:
        # Split the full text into overlapping chunks for model processing
        chunks = chunk_text(
            raw_text,
            chunk_size=int(chunk_size),
            overlap_words=int(overlap),
        )
        st.write(t["chunk_count"].format(n=len(chunks)))

        # Show a preview of the first chunk so the user can verify the split
        if len(chunks) > 0:
            st.text_area(t["first_chunk"], chunks[0].text[:2500], height=220)
    else:
        chunks = []


# ── RIGHT COLUMN: SUMMARIZE + QUIZ ────────────────────────────────────────────
with col_right:
    st.subheader(t["step3"])

    # Streamlit reruns the entire script on every interaction.
    # session_state is used to persist results (summaries, quiz) across reruns.
    if "final_summary" not in st.session_state:
        st.session_state.final_summary = ""
    if "quiz_text" not in st.session_state:
        st.session_state.quiz_text = ""
    if "chunk_summaries" not in st.session_state:
        st.session_state.chunk_summaries = []
    if "summarize_done" not in st.session_state:
        # Flag used to show a success message after st.rerun()
        st.session_state.summarize_done = False
    if "quiz_seed" not in st.session_state:
        # Incremented on each quiz generation to produce different questions
        st.session_state.quiz_seed = 0

    # Load (or retrieve from cache) both models
    llm = get_llm(summarizer_model, quiz_model)

    # Two buttons side by side:
    # Summarize → enabled only when chunks exist
    # Generate Quiz → enabled only when a summary exists
    c1, c2 = st.columns([1, 1])
    with c1:
        do_summarize = st.button(t["btn_summarize"], use_container_width=True, disabled=(not chunks))
    with c2:
        do_quiz = st.button(t["btn_quiz"], use_container_width=True, disabled=(not st.session_state.final_summary))

    # Success banner shown after st.rerun() clears the progress bar
    if st.session_state.summarize_done:
        st.success(t["summary_ready"])
        st.session_state.summarize_done = False  # reset flag

    # ── SUMMARIZATION ─────────────────────────────────────────────────────────
    if do_summarize:
        # Limit to the first N chunks as configured by the user
        chunks_to_use = chunks[: int(use_first_n_chunks)]
        chunks_text = [c.text for c in chunks_to_use]
        total_chunks = len(chunks_text)

        # Live chunk-by-chunk progress bar with a status label
        progress_bar = st.progress(0, text=t["spinner_summarize"])
        status_text = st.empty()

        def _on_chunk_done(done: int, total: int) -> None:
            """Called by llm_pipeline after each chunk is summarized."""
            pct = done / total
            label = t["progress_chunk"].format(done=done, total=total)
            progress_bar.progress(pct, text=label)
            status_text.caption(label)

        # Run summarization — the callback drives the progress bar in real time
        sum_res = llm.summarize_chunks(
            chunks_text,
            mode=summary_mode,
            n_sentences_per_chunk=int(n_sentences),
            on_chunk_done=_on_chunk_done,
        )

        # Briefly show 100% before rerun wipes the progress bar
        progress_bar.progress(1.0, text=t["summary_ready"])
        status_text.empty()

        # Persist results in session_state so they survive the rerun
        st.session_state.chunk_summaries = sum_res.chunk_summaries
        st.session_state.final_summary = sum_res.final_summary
        st.session_state.quiz_text = ""         # clear previous quiz
        st.session_state.summarize_done = True  # trigger success banner on next render

        st.rerun()  # refresh page: progress bar disappears, success banner appears

    # ── QUIZ GENERATION ───────────────────────────────────────────────────────
    if do_quiz:
        # Increment seed so each button press produces a different question set
        st.session_state.quiz_seed += 1
        with st.spinner(t["spinner_quiz"]):
            quiz_res = llm.generate_quiz(
                st.session_state.final_summary,
                n_questions=int(n_questions),
                seed=st.session_state.quiz_seed,
            )
            st.session_state.quiz_text = quiz_res.quiz_text
        st.success(t["quiz_ready"])

    # ── OUTPUT AREAS ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader(t["final_summary"])
    st.text_area("final_summary", st.session_state.final_summary, height=260)

    # Download the summary as a plain-text file
    if st.session_state.final_summary:
        st.download_button(
            t["download_summary"],
            data=st.session_state.final_summary.encode("utf-8"),
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    st.subheader(t["quiz_section"])
    st.text_area("quiz_text", st.session_state.quiz_text, height=260)

    # Download the quiz as a plain-text file
    if st.session_state.quiz_text:
        st.download_button(
            t["download_quiz"],
            data=st.session_state.quiz_text.encode("utf-8"),
            file_name="quiz.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── DEBUG: PER-CHUNK SUMMARIES ────────────────────────────────────────────
    # Expander for developers to inspect each chunk's individual summary
    with st.expander(t["chunk_debug"]):
        for i, cs in enumerate(st.session_state.chunk_summaries, start=1):
            st.markdown(t["chunk_debug_item"].format(i=i))
            st.write(cs)
            st.divider()