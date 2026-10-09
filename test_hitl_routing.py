from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from langgraph_backend import route_after_chat, human_review_node


# builds a fake model reply that asks for the given tools, in order, without calling Gemini
def _reply_asking_for(*tool_names):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {"query": "x"}, "id": f"call-{i}"}
            for i, name in enumerate(tool_names)
        ],
    )


# web search alone must go to human review (this already worked before the fix)
def test_single_web_search_needs_approval():
    state = {"messages": [_reply_asking_for("duckduckgo_search")], "blocked": False}
    assert route_after_chat(state) == "human_review_node"


# web search that is NOT the first tool call must still go to human review (this is the bug)
def test_web_search_second_in_parallel_calls_still_needs_approval():
    state = {"messages": [_reply_asking_for("rag_tool", "duckduckgo_search")], "blocked": False}
    assert route_after_chat(state) == "human_review_node"


# tools that need no approval go straight to the tool node
def test_document_search_alone_skips_approval():
    state = {"messages": [_reply_asking_for("rag_tool")], "blocked": False}
    assert route_after_chat(state) == "tools"


# the approval card must show the web search's own query, even when another tool was requested first
def test_approval_card_shows_the_search_args_not_the_first_call():
    reply = _reply_asking_for("rag_tool", "duckduckgo_search")
    reply.tool_calls[1]["args"] = {"query": "latest OSHA rule"}
    with patch("langgraph_backend.interrupt", return_value="approve") as fake_interrupt:
        human_review_node({"messages": [reply], "blocked": False})
    payload = fake_interrupt.call_args.args[0]
    assert payload["tool_name"] == "duckduckgo_search"
    assert payload["tool_args"] == {"query": "latest OSHA rule"}


# on reject, every tool call gets its own ToolMessage, so the conversation stays valid for Gemini
def test_reject_answers_every_tool_call():
    reply = _reply_asking_for("rag_tool", "duckduckgo_search")
    with patch("langgraph_backend.interrupt", return_value="reject"):
        command = human_review_node({"messages": [reply], "blocked": False})
    answers = command.update["messages"]
    assert command.goto == "chat_node"
    assert all(isinstance(m, ToolMessage) for m in answers)
    assert sorted(m.tool_call_id for m in answers) == ["call-0", "call-1"]
    search_answer = next(m for m in answers if m.name == "duckduckgo_search")
    assert "declined" in search_answer.content
