import os
import tempfile
import torch
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.chunking import chunk_text
from src.llm_pipeline import LLMService
from src.i18n import T

# ── Language selection (must be before set_page_config to use the right title) ──
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Language toggle rendered at the very top of the sidebar via st.sidebar.radio
_lang_choice = st.sidebar.radio(
    "Dil / Language",
    options=["English", "Türkçe"],
    horizontal=True,
    index=0 if st.session_state.lang == "en" else 1,
    key="_lang_radio",
)
st.session_state.lang = "en" if _lang_choice == "English" else "tr"
t = T[st.session_state.lang]

# set_page_config MUST be the very first Streamlit call
st.set_page_config(page_title=t["page_title"], layout="wide")

st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("GPU:", torch.cuda.get_device_name(0))

st.title(t["title"])
st.caption(t["caption"])

with st.sidebar:
    st.header(t["settings"])

    summarizer_model = st.selectbox(
        t["summarizer_model"],
        options=[
            "mukayese/mt5-base-turkish-summarization",
            "ozcangundes/mt5-small-turkish-summarization",
        ],
        index=0,
    )

    quiz_model = st.selectbox(
        t["quiz_model"],
        options=[
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=0,
    )

    max_pages = st.number_input(t["max_pages"], min_value=1, max_value=500, value=10, step=1)

    chunk_size = st.slider(t["chunk_size"], min_value=600, max_value=2000, value=1200, step=100)
    overlap = st.slider(t["overlap"], min_value=0, max_value=80, value=30, step=5)

    use_first_n_chunks = st.number_input(t["first_n_chunks"], min_value=1, max_value=50, value=6, step=1)

    st.divider()
    st.markdown(f"**{t['summary_method']}**")
    summary_mode_label = st.radio(
        label=t["summary_method"],
        options=[t["extractive_label"], t["abstractive_label"]],
        index=0,
        label_visibility="collapsed",
        help=t["extractive_help"],
    )
    summary_mode = "extractive" if summary_mode_label == t["extractive_label"] else "abstractive"

    n_sentences = st.slider(
        t["sentences_per_chunk"],
        min_value=2, max_value=10, value=5, step=1,
        disabled=(summary_mode != "extractive"),
        help=t["sentences_help"],
    )

    n_questions = st.slider(
        t["n_questions"],
        min_value=3, max_value=15, value=5, step=1,
    )

    st.divider()
    st.write(t["tip"])


@st.cache_resource
def get_llm(summarizer_model: str, quiz_model: str) -> LLMService:
    return LLMService(summarizer_model=summarizer_model, quiz_model=quiz_model)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader(t["step1"])

    uploaded = st.file_uploader(t["upload_label"], type=["pdf"])
    pasted_text = st.text_area(t["paste_label"], height=220, placeholder=t["paste_placeholder"])

    show_extracted_text = st.checkbox(t["show_text"], value=False)

    if uploaded is not None:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            extract_res = extract_text_from_pdf(tmp_path, max_pages=int(max_pages))
            raw_text = extract_res.text

            st.success(t["extracted_ok"].format(pages=extract_res.page_count, chars=extract_res.char_count))

            if show_extracted_text:
                st.text_area(t["extracted_preview"], raw_text[:8000], height=250)

        except Exception as exc:
            st.error(f"{t.get('pdf_error', 'PDF okunamadı')}: {exc}")
            raw_text = ""

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        raw_text = pasted_text.strip()
        if raw_text:
            st.info(t["pasted_ok"].format(chars=len(raw_text)))

    st.divider()
    st.subheader(t["step2"])

    if raw_text:
        chunks = chunk_text(
            raw_text,
            chunk_size=int(chunk_size),
            overlap_words=int(overlap),
        )
        st.write(t["chunk_count"].format(n=len(chunks)))

        if len(chunks) > 0:
            st.text_area(t["first_chunk"], chunks[0].text[:2500], height=220)
    else:
        chunks = []


with col_right:
    st.subheader(t["step3"])

    if "final_summary" not in st.session_state:
        st.session_state.final_summary = ""
    if "quiz_text" not in st.session_state:
        st.session_state.quiz_text = ""
    if "chunk_summaries" not in st.session_state:
        st.session_state.chunk_summaries = []
    if "summarize_done" not in st.session_state:
        st.session_state.summarize_done = False
    if "quiz_seed" not in st.session_state:
        st.session_state.quiz_seed = 0

    llm = get_llm(summarizer_model, quiz_model)

    c1, c2 = st.columns([1, 1])
    with c1:
        do_summarize = st.button(t["btn_summarize"], use_container_width=True, disabled=(not chunks))
    with c2:
        do_quiz = st.button(t["btn_quiz"], use_container_width=True, disabled=(not st.session_state.final_summary))

    if st.session_state.summarize_done:
        st.success(t["summary_ready"])
        st.session_state.summarize_done = False

    if do_summarize:
        with st.spinner(t["spinner_summarize"]):
            chunks_to_use = chunks[: int(use_first_n_chunks)]
            chunks_text = [c.text for c in chunks_to_use]

            sum_res = llm.summarize_chunks(
                chunks_text,
                mode=summary_mode,
                n_sentences_per_chunk=int(n_sentences),
            )
            st.session_state.chunk_summaries = sum_res.chunk_summaries
            st.session_state.final_summary = sum_res.final_summary
            st.session_state.quiz_text = ""
            st.session_state.summarize_done = True

        st.rerun()

    if do_quiz:
        st.session_state.quiz_seed += 1
        with st.spinner(t["spinner_quiz"]):
            quiz_res = llm.generate_quiz(
                st.session_state.final_summary,
                n_questions=int(n_questions),
                seed=st.session_state.quiz_seed,
            )
            st.session_state.quiz_text = quiz_res.quiz_text
        st.success(t["quiz_ready"])

    st.divider()
    st.subheader(t["final_summary"])
    st.text_area("final_summary", st.session_state.final_summary, height=260)

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

    if st.session_state.quiz_text:
        st.download_button(
            t["download_quiz"],
            data=st.session_state.quiz_text.encode("utf-8"),
            file_name="quiz.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander(t["chunk_debug"]):
        for i, cs in enumerate(st.session_state.chunk_summaries, start=1):
            st.markdown(t["chunk_debug_item"].format(i=i))
            st.write(cs)
            st.divider()