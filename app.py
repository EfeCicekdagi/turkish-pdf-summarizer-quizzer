import tempfile
import torch
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.chunking import chunk_text
from src.llm_pipeline import LLMService

# set_page_config MUST be the very first Streamlit call
st.set_page_config(page_title="Turkish PDF Summarizer + Quiz", layout="wide")

st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("GPU:", torch.cuda.get_device_name(0))

st.title("\U0001f4c4 Turkish PDF Summarizer + Quiz Generator")
st.caption("Upload a PDF \u2192 split into chunks \u2192 summarize \u2192 generate 5 questions")

with st.sidebar:
    st.header("Settings")

    summarizer_model = st.selectbox(
        "Summarizer Model (TR)",
        options=[
            "mukayese/mt5-base-turkish-summarization",
            "ozcangundes/mt5-small-turkish-summarization",
        ],
        index=0,
    )

    quiz_model = st.selectbox(
        "Quiz Model",
        options=[
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=0,
    )

    max_pages = st.number_input("Max pages to extract (for testing)", min_value=1, max_value=500, value=10, step=1)

    chunk_size = st.slider("Chunk size (characters)", min_value=600, max_value=2000, value=1200, step=100)
    overlap = st.slider("Overlap (word count)", min_value=0, max_value=80, value=30, step=5)

    use_first_n_chunks = st.number_input("Quick demo: use first N chunks", min_value=1, max_value=50, value=6, step=1)

    st.divider()
    st.markdown("**Summary Method**")
    summary_mode_label = st.radio(
        label="Summary Method",
        options=[
            "Extractive \u2014 reliable, no hallucination",
            "Abstractive (mT5) \u2014 fluent but may hallucinate",
        ],
        index=0,
        label_visibility="collapsed",
        help="Extractive: selects sentences from the original text.\nAbstractive: mT5 generates new sentences.",
    )
    summary_mode = "extractive" if summary_mode_label.startswith("E") else "abstractive"

    n_sentences = st.slider(
        "Sentences per chunk",
        min_value=2, max_value=10, value=5, step=1,
        disabled=(summary_mode != "extractive"),
        help="Only used in Extractive mode.",
    )

    st.divider()
    st.write("Tip: For large PDFs, keep `max pages` and `first N chunks` small.")


@st.cache_resource
def get_llm(summarizer_model: str, quiz_model: str) -> LLMService:
    return LLMService(summarizer_model=summarizer_model, quiz_model=quiz_model)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) Upload PDF or paste text")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    pasted_text = st.text_area("...or paste text here", height=220, placeholder="If you don't have a PDF, you can paste text here.")

    show_extracted_text = st.checkbox("Show extracted text", value=False)

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        extract_res = extract_text_from_pdf(tmp_path, max_pages=int(max_pages))
        raw_text = extract_res.text

        st.success(f"Text extracted \u2705  Pages: {extract_res.page_count} | Characters: {extract_res.char_count}")

        if show_extracted_text:
            st.text_area("Extracted text (preview)", raw_text[:8000], height=250)

    else:
        raw_text = pasted_text.strip()
        if raw_text:
            st.info(f"Pasted text received \u2705  Characters: {len(raw_text)}")

    st.divider()
    st.subheader("2) Chunking")

    if raw_text:
        chunks = chunk_text(
            raw_text,
            chunk_size=int(chunk_size),
            overlap_words=int(overlap),
        )
        st.write(f"Number of chunks: **{len(chunks)}**")

        if len(chunks) > 0:
            st.text_area("First chunk preview", chunks[0].text[:2500], height=220)
    else:
        chunks = []


with col_right:
    st.subheader("3) Summarize + Generate Quiz")

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
        do_summarize = st.button("\U0001f9e0 Summarize", use_container_width=True, disabled=(not chunks))
    with c2:
        do_quiz = st.button("\U0001f4dd Generate Quiz", use_container_width=True, disabled=(not st.session_state.final_summary))

    if st.session_state.summarize_done:
        st.success("Summary ready \u2705")
        st.session_state.summarize_done = False

    if do_summarize:
        with st.spinner("Summarizing chunks and building final summary..."):
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
        with st.spinner("Generating quiz..."):
            quiz_res = llm.generate_quiz(
                st.session_state.final_summary,
                n_questions=5,
                seed=st.session_state.quiz_seed,
            )
            st.session_state.quiz_text = quiz_res.quiz_text
        st.success("Quiz ready \u2705")

    st.divider()
    st.subheader("Final Summary")
    st.text_area("final_summary", st.session_state.final_summary, height=260)

    if st.session_state.final_summary:
        st.download_button(
            "\u2b07\ufe0f Download summary (txt)",
            data=st.session_state.final_summary.encode("utf-8"),
            file_name="summary_tr.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()
    st.subheader("Quiz")
    st.text_area("quiz_text", st.session_state.quiz_text, height=260)

    if st.session_state.quiz_text:
        st.download_button(
            "\u2b07\ufe0f Download quiz (txt)",
            data=st.session_state.quiz_text.encode("utf-8"),
            file_name="quiz_tr.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Show chunk summaries (debug)"):
        for i, cs in enumerate(st.session_state.chunk_summaries, start=1):
            st.markdown(f"**Chunk Summary {i}**")
            st.write(cs)
            st.divider()