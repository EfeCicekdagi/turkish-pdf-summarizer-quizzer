from pdf_utils import extract_text_from_pdf
from chunking import chunk_text
from llm_pipeline import LLMService

res = extract_text_from_pdf("examples/sample_input.pdf", max_pages=5)
chunks = chunk_text(res.text, chunk_size=1200, overlap=150)

# sadece ilk 3 chunk ile hızlı test
chunks_text = [c.text for c in chunks[:3]]

llm = LLMService(model_name="google/flan-t5-base")
sum_res = llm.summarize_chunks(chunks_text)

print("=== FINAL SUMMARY ===")
print(sum_res.final_summary)

quiz = llm.generate_quiz(sum_res.final_summary, n_questions=5)
print("\n=== QUIZ ===")
print(quiz.quiz_text)