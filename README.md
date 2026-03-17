# Lightweight Academic-Paper RAG System

This project is built for a course demo. It imports local academic papers, builds a vector index, and provides retrieval-augmented question answering through a web interface.

## 1) Environment

- Python 3.10+
- A virtual environment is recommended (`venv` or `conda`)
- If you use `UnstructuredPDFLoader(strategy="hi_res")` on Windows, install:
  - Tesseract OCR
  - Poppler

## 2) Install Dependencies

```bash
pip install -r requirements.txt
```

## 3) Configure LLM API Key

Create or edit a `.env` file in the project root:

```env
TONGYI_API_KEY=your_tongyi_api_key
```

## 4) Prepare Paper Data

Put your paper files into `knowledge_base/`.

Supported formats in `indexing.py`:

- `.pdf`
- `.md`
- `.txt`
- `.docx`

The script automatically scans `knowledge_base/` and builds the index from all supported files.

## 5) Build the Vector Store

```bash
python indexing.py
```

Notes:

- The script prints the number of loaded files and parsed document chunks.
- `vectorstore/` is rebuilt on each indexing run to avoid duplicate accumulation.

## 6) Start the QA Service

```bash
python app.py
```

Then open:

- [http://127.0.0.1:8089](http://127.0.0.1:8089)

## 7) Quick Reproduction

```bash
pip install -r requirements.txt
python indexing.py
python app.py
```

## 8) Troubleshooting

- API key errors: verify `TONGYI_API_KEY` in `.env`
- PDF parsing issues: make sure Tesseract/Poppler are installed and callable from terminal
- Retrieval quality issues: rerun `python indexing.py` to rebuild `vectorstore/`
