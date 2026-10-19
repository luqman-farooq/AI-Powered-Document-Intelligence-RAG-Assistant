# Nexa AI — Local Document Intelligence & RAG Assistant

A local-first Streamlit assistant that supports general web-assisted research and question answering over uploaded PDF/TXT documents.

## Features

- Streamlit chat UI with multiple chats
- PDF and TXT ingestion
- Text chunking and local embeddings through Ollama
- FAISS vector index stored locally
- Document question answering with source/page references
- Web and news search through DDGS, with relevance scoring and optional date-range filtering
- Local chat generation through Ollama; no hosted LLM API key is required

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.11 recommended
- Ollama installed and running
- Ollama models: `qwen2.5:3b` and `nomic-embed-text`

Pull the models if needed:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

## Setup (Windows PowerShell)

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

If PowerShell blocks activation, use the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

## Configuration

Defaults are local and do not require an API key. Optional `.env` values:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=qwen2.5:3b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

The chat model and embedding model must exist in `ollama list`.

## How to use

1. Open the app in your browser.
2. Select **General AI** to ask questions using web/news search evidence.
3. Select **Chat with documents** or attach PDF/TXT files in the chat input.
4. The app indexes uploaded documents into `data/vector_store`.
5. Ask questions about the indexed text and review the displayed source references.

Only text extractable from PDFs can be searched. Scanned/image-only PDFs may require OCR, which is not included.

## Tests

```powershell
.\.venv\Scripts\python.exe -m compileall -q app src tests
.\.venv\Scripts\python.exe -m pytest -q
```

## Architecture

PDF/TXT → text extraction → chunking → Ollama embeddings → local FAISS index → retrieval → local Ollama answer + document source references.

General AI mode → DDGS text/news results → relevance/date filtering → local Ollama synthesis → clickable web sources.

## Safety and limitations

- Web search snippets are search-engine evidence, not guaranteed ground truth. Verify important claims against the linked pages.
- Publication-date filtering depends on date metadata supplied by the search provider. Some valid results may be excluded if their dates are missing.
- The "official" query filter only includes domains classified by this project's allowlist; the list is intentionally limited and may miss legitimate official organizations.
- The local FAISS loader uses pickle-backed metadata. Only load an index created by this trusted local application; never replace it with an untrusted index.
- This is a portfolio prototype, not a production-secured multi-user service.

## Project layout

```text
app/app.py
src/config.py
src/document_loader.py
src/text_splitter.py
src/embeddings.py
src/vector_store.py
src/retriever.py
src/rag_pipeline.py
src/web_search.py
tests/test_rag_components.py
requirements.txt
.env.example
```
