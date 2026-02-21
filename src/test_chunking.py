from pdf_utils import extract_text_from_pdf
from chunking import chunk_text

res = extract_text_from_pdf("examples/sample_input.pdf", max_pages=5)
chunks = chunk_text(res.text, chunk_size=1200, overlap=150)

print("Total chars:", res.char_count)
print("Chunk count:", len(chunks))
print("First chunk chars:", chunks[0].char_len)
print("--- FIRST CHUNK PREVIEW ---")
print(chunks[0].text[:600])