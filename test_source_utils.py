import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from source_utils import extract_sources, format_sources


# builds a fake rag_tool result message shaped like the real ones the chatbot stores
def _rag_message(pages, source_file="doc.pdf"):
    payload = {
        "query": "q",
        "context": ["chunk"] * len(pages),
        "pages": pages,
        "source_file": source_file,
        "cache_hit": False,
    }
    return ToolMessage(content=json.dumps(payload), name="rag_tool", tool_call_id="t1")


# verifies page numbers are converted from 0-based to 1-based, de-duplicated, and sorted
def test_pages_are_one_based_deduplicated_and_sorted():
    messages = [HumanMessage(content="q"), _rag_message([7, 2, 7, 0])]
    assert extract_sources(messages) == {"doc.pdf": [1, 3, 8]}


# verifies a retrieval from an earlier question is not attributed to the latest answer
def test_only_latest_turn_is_used():
    messages = [
        HumanMessage(content="first"),
        _rag_message([10], source_file="old.pdf"),
        AIMessage(content="old answer"),
        HumanMessage(content="second"),
        _rag_message([4], source_file="new.pdf"),
        AIMessage(content="new answer"),
    ]
    assert extract_sources(messages) == {"new.pdf": [5]}


# verifies several rag_tool calls in one turn are merged into one source list
def test_multiple_calls_in_one_turn_are_merged():
    messages = [HumanMessage(content="q"), _rag_message([1]), _rag_message([5, 1])]
    assert extract_sources(messages) == {"doc.pdf": [2, 6]}


# verifies a turn with no document retrieval yields no sources
def test_no_rag_tool_means_no_sources():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    assert extract_sources(messages) == {}


# verifies other tools' results are ignored, even if their text looks like JSON
def test_other_tools_are_ignored():
    other = ToolMessage(content=json.dumps({"pages": [3]}), name="calculator", tool_call_id="t2")
    assert extract_sources([HumanMessage(content="q"), other]) == {}


# verifies an unreadable rag_tool message is skipped instead of crashing the UI
def test_bad_json_is_skipped():
    broken = ToolMessage(content="not json", name="rag_tool", tool_call_id="t3")
    assert extract_sources([HumanMessage(content="q"), broken]) == {}


# verifies the display line format, and that an empty result produces no text at all
def test_format_sources():
    assert format_sources({"doc.pdf": [3, 8]}) == "Sources retrieved: doc.pdf (PDF pages 3, 8)"
    assert format_sources({}) == ""