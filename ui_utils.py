from langchain_core.messages import HumanMessage

# longest sidebar title before it is cut with an ellipsis
TITLE_MAX_LEN = 40
# shown for a conversation with no question yet
EMPTY_TITLE = "New chat"


# turns a conversation's first question into a short sidebar title, instead of showing the raw thread UUID
def conversation_title(messages, max_len=TITLE_MAX_LEN):
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            text = " ".join(str(content).split())
            if not text:
                continue
            return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"
    return EMPTY_TITLE
