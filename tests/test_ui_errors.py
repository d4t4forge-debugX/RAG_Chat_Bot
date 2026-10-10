from unittest.mock import MagicMock, patch

from pathlib import Path

from streamlit.testing.v1 import AppTest

# AppTest resolves paths relative to this test file, so point it at the app one folder up
APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_frontend.py")

DAILY_QUOTA = (
    "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "limit: 500 Please retry in 8h57m50.969470679s. "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
)


# verifies a model failure during a normal chat turn becomes a friendly assistant message instead of a crash
def test_failed_chat_turn_shows_friendly_message():
    fake_chatbot = MagicMock()
    fake_chatbot.stream.side_effect = RuntimeError(DAILY_QUOTA)

    with patch("langgraph_backend.chatbot", fake_chatbot):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        at.chat_input[0].set_value("hello").run()

    assert not at.exception
    last = at.session_state["message_history"][-1]
    assert last["role"] == "assistant"
    assert "daily" in last["content"]
    assert "RESOURCE_EXHAUSTED" not in last["content"]


# verifies a failure while resuming an approval shows a friendly error and keeps the approval card for a retry
def test_failed_resume_keeps_approval_card():
    fake_chatbot = MagicMock()
    fake_chatbot.invoke.side_effect = RuntimeError(DAILY_QUOTA)

    with patch("langgraph_backend.chatbot", fake_chatbot):
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.session_state["pending_interrupt"] = {
            "question": "Approve this tool call?",
            "tool_name": "duckduckgo_search",
            "tool_args": {"query": "x"},
        }
        at.run()
        approve = [b for b in at.button if "Approve" in b.label][0]
        approve.click().run()

    assert not at.exception
    assert any("daily" in e.value for e in at.error)
    assert at.session_state["pending_interrupt"] is not None