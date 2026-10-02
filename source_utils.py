import json

from langchain_core.messages import HumanMessage, ToolMessage


# collects {source file: sorted page numbers} from the rag_tool results in the latest turn only (stops at the last user message)
def extract_sources(messages):
    """
    Walks backwards through the message list until the latest HumanMessage.
    Every rag_tool ToolMessage on the way is a retrieval done for this
    question. PyPDFLoader numbers pages from 0, so we add 1 to show the
    page position a person would see in a PDF viewer.
    """
    pages_by_file = {}
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and msg.name == "rag_tool":
            try:
                data = json.loads(msg.content)
            except (TypeError, json.JSONDecodeError):
                continue
            filename = data.get("source_file") or "document"
            for page in data.get("pages", []):
                if page is not None:
                    pages_by_file.setdefault(filename, set()).add(page + 1)
    return {filename: sorted(pages) for filename, pages in pages_by_file.items()}


# turns the sources dict into one short display line, or an empty string when there is nothing to show
def format_sources(sources):
    if not sources:
        return ""
    parts = [
        f"{filename} (PDF pages {', '.join(str(p) for p in pages)})"
        for filename, pages in sources.items()
    ]
    return "Sources retrieved: " + "; ".join(parts)