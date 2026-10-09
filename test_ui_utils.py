from langchain_core.messages import AIMessage, HumanMessage

from ui_utils import conversation_title, EMPTY_TITLE


# the title is the first question the user asked, not a later one or the assistant's reply
def test_title_is_first_user_question():
    messages = [HumanMessage(content="What is lockout?"), AIMessage(content="It is..."), HumanMessage(content="Second")]
    assert conversation_title(messages) == "What is lockout?"


# long questions are cut to the limit and end with an ellipsis
def test_long_question_is_truncated():
    title = conversation_title([HumanMessage(content="word " * 30)], max_len=20)
    assert len(title) <= 20 and title.endswith("…")


# a conversation with no question yet gets a placeholder instead of an empty button
def test_empty_conversation_gets_placeholder():
    assert conversation_title([]) == EMPTY_TITLE


# line breaks and repeated spaces collapse to single spaces so the button stays on one line
def test_whitespace_is_collapsed():
    assert conversation_title([HumanMessage(content="  What   is\n\nX?  ")]) == "What is X?"
