from unittest.mock import patch, MagicMock
from langgraph_backend import is_query_appropriate


# builds a fake LLM response object with just a .content attribute, matching what llm.invoke() actually returns
def _mock_response(text):
    """Builds a fake LLM response object with just a .content attribute,
    matching the shape is_query_appropriate() expects back from llm.invoke()."""
    mock_response = MagicMock()
    mock_response.content = text
    return mock_response


# verifies a clean "YES" response from the LLM is correctly parsed as an appropriate query
def test_appropriate_query_returns_true():
    with patch("langgraph_backend.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response("YES")
        assert is_query_appropriate("What is supervised learning?") is True


# verifies a clean "NO" response from the LLM is correctly parsed as an inappropriate query
def test_inappropriate_query_returns_false():
    with patch("langgraph_backend.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response("NO")
        assert is_query_appropriate("some abusive message") is False


# verifies the substring "yes" check still works even when it's lowercase and embedded in a longer sentence
def test_lowercase_yes_still_parsed_correctly():
    with patch("langgraph_backend.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response("yes, this is fine")
        assert is_query_appropriate("a reasonable question") is True


# verifies the function correctly handles list-of-content-block responses, not just plain strings
def test_list_content_response_handled():
    """Some LLM responses come back as a list of content blocks instead of
    a plain string (same shape your extract_text() helpers already handle
    elsewhere in the app) — confirm is_query_appropriate() copes with that too."""
    with patch("langgraph_backend.llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response([{"type": "text", "text": "YES"}])
        assert is_query_appropriate("another question") is True