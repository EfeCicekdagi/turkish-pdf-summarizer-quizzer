import tempfile
import torch
import streamlit as st

from src.pdf_utils import extract_text_from_pdf
from src.chunking import chunk_text
from src.llm_pipeline import LLMService

# set_page_config MUST be the very first Streamlit call
st.set_page_config(page_title="PDF Summarizer + Quiz (TR)", layout="wide")

st.sidebar.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.sidebar.write("GPU:", torch.cuda.get_device_name(0))

st.title("📄 Türkçe PDF Özetleyici + Quiz Üretici (Hugging Face)")
st.caption("PDF yükle → chunk’lara böl → özetle → 5 soru üret")

with st.sidebar:
    st.header("Ayarlar")

    summarizer_model = st.selectbox(
        "Özet Modeli (TR)",
        options=[
            "mukayese/mt5-base-turkish-summarization",
            "ozcangundes/mt5-small-turkish-summarization",
        ],
        index=0,
    )

    quiz_model = st.selectbox(
        "Quiz Modeli",
        options=[
            "google/flan-t5-base",
            "google/flan-t5-large",
        ],
        index=0,
    )

    max_pages = st.number_input("PDF için max sayfa (test için)", min_value=1, max_value=500, value=10, step=1)

    chunk_size = st.slider("Chunk boyutu (karakter)", min_value=600, max_value=2000, value=1200, step=100)
    overlap = st.slider("Overlap (kelime sayısı)", min_value=0, max_value=80, value=30, step=5)

    use_first_n_chunks = st.number_input("Hızlı demo: ilk N chunk ile çalış", min_value=1, max_value=50, value=6, step=1)

    st.divider()
    st.write("İpucu: Büyük PDF’lerde `max_pages` ve `ilk N chunk` değerini küçük tut.")


@st.cache_resource
def get_llm(summarizer_model: str, quiz_model: str) -> LLMService:
    return LLMService(summarizer_model=summarizer_model, quiz_model=quiz_model)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1) PDF yükle veya metin yapıştır")

    uploaded = st.file_uploader("PDF yükle", type=["pdf"])
    pasted_text = st.text_area("...veya buraya metin yapıştır", height=220, placeholder="PDF yoksa buraya metin yapıştırabilirsin.")

    show_extracted_text = st.checkbox("Çıkarılan metni göster", value=False)

    if uploaded is not None:
        # Save uploaded file to a temp path for PyMuPDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        extract_res = extract_text_from_pdf(tmp_path, max_pages=int(max_pages))
        raw_text = extract_res.text

        st.success(f"Metin çıkarıldı ✅  Sayfa: {extract_res.page_count} | Karakter: {extract_res.char_count}")

        if show_extracted_text:
            st.text_area("Çıkarılan metin (preview)", raw_text[:8000], height=250)

    else:
        raw_text = pasted_text.strip()
        if raw_text:
            st.info(f"Yapıştırılan metin alındı ✅  Karakter: {len(raw_text)}")

    st.divider()
    st.subheader("2) Chunking")

    if raw_text:
        chunks = chunk_text(
            raw_text,
            chunk_size=int(chunk_size),
            overlap_words=int(overlap),)
        st.write(f"Chunk sayısı: **{len(chunks)}**")

        if len(chunks) > 0:
            st.text_area("İlk chunk preview", chunks[0].text[:2500], height=220)
    else:
        chunks = []


with col_right:
    st.subheader("3) Özet + Quiz üret")

    if "final_summary" not in st.session_state:
        st.session_state.final_summary = ""
    if "quiz_text" not in st.session_state:
        st.session_state.quiz_text = ""
    if "chunk_summaries" not in st.session_state:
        st.session_state.chunk_summaries = []

    llm = get_llm(summarizer_model, quiz_model)

    c1, c2 = st.columns([1, 1])
    with c1:
        do_summarize = st.button("🧠 Özetle", use_container_width=True, disabled=(not chunks))
    with c2:
        do_quiz = st.button("📝 Quiz Üret", use_container_width=True, disabled=(not st.session_state.final_summary))

    if do_summarize:
        with st.spinner("Chunk’lar özetleniyor ve final özet hazırlanıyor..."):
            chunks_to_use = chunks[: int(use_first_n_chunks)]
            chunks_text = [c.text for c in chunks_to_use]

            sum_res = llm.summarize_chunks(chunks_text)
            st.session_state.chunk_summaries = sum_res.chunk_summaries
            st.session_state.final_summary = sum_res.final_summary
            st.session_state.quiz_text = ""

        st.success("Özet hazır ✅")

    if do_quiz:
        with st.spinner("Quiz üretiliyor..."):
            quiz_res = llm.generate_quiz(st.session_state.final_summary, n_questions=5)
            st.session_state.quiz_text = quiz_res.quiz_text
        st.success("Quiz hazır ✅")

    st.divider()
    st.subheader("Final Özet")
    st.text_area("final_summary", st.session_state.final_summary, height=260)

    if st.session_state.final_summary:
        st.download_button(
            "⬇️ Özeti indir (txt)",
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
            "⬇️ Quiz indir (txt)",
            data=st.session_state.quiz_text.encode("utf-8"),
            file_name="quiz_tr.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with st.expander("Chunk özetlerini göster (debug)"):
        for i, cs in enumerate(st.session_state.chunk_summaries, start=1):
            st.markdown(f"**Chunk Özet {i}**")
            st.write(cs)
            st.divider()