from src.vector_store import get_vector_store


def retrieve_documents(question: str, k: int = 8):
    question = (question or "").strip()
    if not question:
        return []

    vector_store = get_vector_store()
    results = []

    # Similarity search is the primary path; MMR adds diversity if supported.
    results.extend(vector_store.similarity_search(question, k=k))
    try:
        results.extend(
            vector_store.max_marginal_relevance_search(
                question, k=min(k, 8), fetch_k=max(20, k * 2), lambda_mult=0.7
            )
        )
    except (AttributeError, NotImplementedError, ValueError):
        pass

    unique_documents = []
    seen = set()
    for doc in results:
        content = doc.page_content.strip()
        key = (content, doc.metadata.get("source"), doc.metadata.get("page"))
        if content and key not in seen:
            seen.add(key)
            unique_documents.append(doc)
        if len(unique_documents) >= k:
            break
    return unique_documents


def format_context(documents):
    context_parts = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        location = f"{source} - page {page + 1}" if isinstance(page, int) else source
        content = doc.page_content.strip()
        if content:
            context_parts.append(
                f"[Document section {index}]\nSource: {location}\nContent:\n{content}"
            )
    return "\n\n---\n\n".join(context_parts)
