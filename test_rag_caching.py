from unittest.mock import patch, MagicMock
import pytest
from rag_utils import rag_tool, _QUERY_CACHE, _clear_thread_cache


@pytest.fixture(autouse=True)
def clear_cache_before_each_test():
    """
    _QUERY_CACHE is a module-level dict shared across all tests in this
    file. Without resetting it, a cached entry from one test could make a
    later test pass (or fail) for the wrong reason. autouse=True means
    this runs automatically before every test below, no need to reference
    it explicitly in each test signature.
    """
    _QUERY_CACHE.clear()
    yield
    _QUERY_CACHE.clear()


def _make_fake_retriever(page_content="Fake chunk content", page=1):
    """A minimal stand-in for a real EnsembleRetriever: .invoke(query)
    returns a list of fake documents with just page_content and metadata,
    matching what rag_tool actually reads off each result."""
    fake_doc = MagicMock()
    fake_doc.page_content = page_content
    fake_doc.metadata = {"page": page}

    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = [fake_doc]
    return fake_retriever


def test_first_call_is_cache_miss():
    fake_retriever = _make_fake_retriever()
    with patch("rag_utils.get_retriever_for_thread", return_value=fake_retriever):
        result = rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        assert result["cache_hit"] is False
        assert result["context"] == ["Fake chunk content"]
        fake_retriever.invoke.assert_called_once_with("What is X?")


def test_second_identical_call_is_cache_hit():
    fake_retriever = _make_fake_retriever()
    with patch("rag_utils.get_retriever_for_thread", return_value=fake_retriever):
        rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        result = rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})

        assert result["cache_hit"] is True
        # The retriever should only have been hit once — the second call
        # should be served entirely from _QUERY_CACHE.
        fake_retriever.invoke.assert_called_once()


def test_cache_key_is_case_and_whitespace_insensitive():
    """rag_tool normalizes the cache key via query.strip().lower() — confirm
    a differently-cased/spaced but semantically identical query still hits
    the same cache entry."""
    fake_retriever = _make_fake_retriever()
    with patch("rag_utils.get_retriever_for_thread", return_value=fake_retriever):
        rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        result = rag_tool.invoke({"query": "  WHAT IS X?  ", "thread_id": "thread-1"})

        assert result["cache_hit"] is True
        fake_retriever.invoke.assert_called_once()


def test_different_threads_do_not_share_cache():
    fake_retriever = _make_fake_retriever()
    with patch("rag_utils.get_retriever_for_thread", return_value=fake_retriever):
        rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        result = rag_tool.invoke({"query": "What is X?", "thread_id": "thread-2"})

        assert result["cache_hit"] is False
        assert fake_retriever.invoke.call_count == 2


def test_clear_thread_cache_removes_only_that_threads_entries():
    fake_retriever = _make_fake_retriever()
    with patch("rag_utils.get_retriever_for_thread", return_value=fake_retriever):
        rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        rag_tool.invoke({"query": "What is Y?", "thread_id": "thread-2"})

        _clear_thread_cache("thread-1")

        result_thread_1 = rag_tool.invoke({"query": "What is X?", "thread_id": "thread-1"})
        result_thread_2 = rag_tool.invoke({"query": "What is Y?", "thread_id": "thread-2"})

        assert result_thread_1["cache_hit"] is False  # cache was cleared, re-fetched
        assert result_thread_2["cache_hit"] is True    # untouched, still cached