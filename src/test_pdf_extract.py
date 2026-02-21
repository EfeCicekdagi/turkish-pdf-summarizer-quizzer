import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from pdf_utils import extract_text_from_pdf

result = extract_text_from_pdf("examples/sample_input.pdf", max_pages=2)
print("pages:", result.page_count)
print("chars:", result.char_count)
print(result.text[:800])