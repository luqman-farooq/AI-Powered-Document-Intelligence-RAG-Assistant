# System Architecture

## High-Level Flow

```text
                 +------------------+
                 | PDF / TXT Files  |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Document Loader  |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Text Chunking    |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Embedding Model  |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 | Chroma Vector DB |
                 +--------+---------+
                          ^
                          |
User Question ---> Similarity Search
                          |
                          v
                 +------------------+
                 | Relevant Context |
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |       LLM        |
                 +--------+---------+
                          |
                          v
                    Final Answer
```

## Components

1. **Document Loader** extracts text from PDF/TXT files.
2. **Text Splitter** creates manageable chunks.
3. **Embeddings** convert chunks into vectors.
4. **Chroma** stores vectors for similarity search.
5. **Retriever** finds relevant chunks for a question.
6. **LLM** generates an answer using retrieved context.
7. **Streamlit** provides the user interface.

## RAG Principle

The LLM is not asked to answer from memory alone. Relevant document context is retrieved first and included in the prompt. This reduces unsupported answers and makes the application useful for private document collections.
