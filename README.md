# 📄 Turkish PDF Summarizer + Quiz Generator

> **Türkçe PDF belgelerini özetle ve quiz üret — yerel modeller, internet bağlantısı gerekmez.**  
> Summarize Turkish PDF documents and generate quizzes — local models, no internet required.

---

## 🇹🇷 Türkçe

### Nedir?

Bu uygulama, Türkçe PDF dosyalarını (ya da yapıştırılan metinleri) otomatik olarak özetleyen ve içerikten quiz soruları üreten bir **Streamlit** web uygulamasıdır. Tüm işlemler yerel makinenizde çalışır; verileriniz hiçbir sunucuya gönderilmez.

### Özellikler

- 📂 **PDF yükleme veya metin yapıştırma** — esnek giriş yöntemi
- ✂️ **Akıllı chunking** — metni örtüşen parçalara böler
- 🧠 **İki özet modu:**
  - **Çıkarımsal (Extractive):** TF-IDF ile orijinal cümleleri seçer — hızlı, güvenilir, sıfır halüsinasyon
  - **Üretimsel (Abstractive):** mT5 ile yeni cümleler üretir — akıcı ama yavaş
- 📝 **Quiz üretici** — özetten otomatik soru-cevap çifti oluşturur
- 🌐 **İki dil desteği** — Türkçe ve İngilizce arayüz
- ⬇️ **Özet ve quiz indirme** — `.txt` dosyası olarak

### Kullanılan Modeller

| Görev | Model |
|-------|-------|
| Türkçe özetleme | `mukayese/mt5-base-turkish-summarization` *(varsayılan)* |
| Türkçe özetleme (küçük) | `ozcangundes/mt5-small-turkish-summarization` |
| Quiz modeli | `google/flan-t5-base` *(varsayılan)* |
| Quiz modeli (büyük) | `google/flan-t5-large` |

### Kurulum

```bash
# 1. Depoyu klonla
git clone https://github.com/EfeCicekdagi/turkish-pdf-summarizer-quizzer.git
cd turkish-pdf-summarizer-quizzer

# 2. Sanal ortam oluştur (isteğe bağlı ama önerilir)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Bağımlılıkları yükle
pip install -r requirements.txt

# 4. Uygulamayı çalıştır
streamlit run app.py
```

> **GPU desteği:** CUDA kurulu ve uyumlu bir GPU varsa modeller otomatik olarak GPU'da çalışır.

### Kullanım

1. Uygulamayı tarayıcıda aç (`http://localhost:8501`)
2. Sol bölmeden ayarları yapılandır (model seçimi, chunk boyutu, özet yöntemi vb.)
3. PDF yükle veya metin yapıştır
4. **🧠 Özetle** butonuna bas
5. Özet hazır olduktan sonra **📝 Quiz Üret** butonuna bas
6. İstersen özet ve quiz'i `.txt` olarak indir

---

## 🇬🇧 English

### What is it?

A **Streamlit** web application that automatically summarizes Turkish PDF files (or pasted text) and generates quiz questions from the content. Everything runs locally — no data is sent to any external server.

### Features

- 📂 **PDF upload or text paste** — flexible input
- ✂️ **Smart chunking** — splits text into overlapping segments
- 🧠 **Two summarization modes:**
  - **Extractive:** picks original sentences via TF-IDF — fast, reliable, zero hallucination
  - **Abstractive:** generates new sentences with mT5 — fluent but slower
- 📝 **Quiz generator** — creates Q&A pairs automatically from the summary
- 🌐 **Bilingual UI** — Turkish and English interface
- ⬇️ **Download summary & quiz** — as `.txt` files

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/EfeCicekdagi/turkish-pdf-summarizer-quizzer.git
cd turkish-pdf-summarizer-quizzer

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

### Usage

1. Open the app in your browser (`http://localhost:8501`)
2. Configure settings in the sidebar (model selection, chunk size, summary method, etc.)
3. Upload a PDF or paste text
4. Click **🧠 Summarize**
5. Once the summary is ready, click **📝 Generate Quiz**
6. Download the summary and quiz as `.txt` if needed

---

## 🗂️ Project Structure

```
turkish-pdf-summarizer-quizzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── src/
    ├── pdf_utils.py        # PDF text extraction (PyMuPDF)
    ├── chunking.py         # Text chunking logic
    ├── extractive.py       # TF-IDF extractive summarizer
    ├── llm_pipeline.py     # LLM service (mT5 + FLAN-T5)
    ├── quiz_generator.py   # Template-based quiz generator
    ├── prompts.py          # Prompt builders
    ├── postprocess.py      # Output normalization
    └── i18n.py             # UI translations (TR / EN)
```

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `transformers` | mT5 & FLAN-T5 models |
| `torch` | Deep learning backend (CPU/GPU) |
| `pymupdf` | PDF text extraction |

## 📄 License

[MIT](LICENSE)