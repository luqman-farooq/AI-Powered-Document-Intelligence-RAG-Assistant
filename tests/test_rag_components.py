from langchain_core.documents import Document

from src.text_splitter import split_documents
from src.retriever import format_context
from src.web_search import requested_date_range, relevance_score, is_trusted, is_official


def test_split_documents():
    docs = [Document(page_content="Python is a programming language. " * 100)]
    chunks = split_documents(docs)
    assert len(chunks) > 1
    assert all(chunk.page_content for chunk in chunks)


def test_format_context():
    docs = [Document(page_content="Machine learning learns patterns.", metadata={"source": "test.txt", "page": 0})]
    context = format_context(docs)
    assert "Machine learning" in context
    assert "test.txt" in context
    assert "page 1" in context


def test_requested_date_range_includes_end_day():
    result = requested_date_range("official AI announcements between October 1 and October 16, 2026")
    assert result is not None
    start, end, _ = result
    assert start.isoformat().startswith("2026-10-01")
    assert end.isoformat().startswith("2026-10-17")


def test_requested_month_range():
    result = requested_date_range("AI news October 2026")
    assert result is not None
    start, end, _ = result
    assert start.isoformat().startswith("2026-10-01")
    assert end.isoformat().startswith("2026-11-01")


def test_relevance_prefers_matching_title():
    assert relevance_score("OpenAI new model release", "OpenAI releases a new model", "Details available") > relevance_score(
        "OpenAI new model release", "Weather forecast", "Some unrelated news"
    )


def test_domain_classification():
    assert is_trusted("https://openai.com/blog/example")
    assert is_official("https://openai.com/blog/example")
    assert not is_official("https://example-news-site.com/article")
