from unittest.mock import patch

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from langgraph_backend import ChatState, handle_tool_failure, tool_node


# builds a tiny graph containing only the real tool node, so it runs with the runtime LangGraph normally provides
def _tool_only_graph():
    g = StateGraph(ChatState)
    g.add_node("tools", tool_node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    return g.compile()


# verifies a crashing web-search tool is reported back as a ToolMessage instead of raising and killing the app
def test_search_failure_becomes_tool_message_instead_of_crashing():
    fake_call = AIMessage(
        content="",
        tool_calls=[{"name": "duckduckgo_search", "args": {"query": "x"}, "id": "call-1"}],
    )
    with patch.object(DuckDuckGoSearchRun, "_run", side_effect=RuntimeError("no network")):
        result = _tool_only_graph().invoke({"messages": [fake_call], "blocked": False})

    reply = result["messages"][-1]
    assert isinstance(reply, ToolMessage)
    assert "could not complete" in reply.content


# verifies the failure message tells the model not to retry, so a dead network can't cause a loop of approval cards
def test_failure_message_tells_the_model_not_to_retry():
    message = handle_tool_failure(RuntimeError("boom"))
    assert "Do not retry" in message