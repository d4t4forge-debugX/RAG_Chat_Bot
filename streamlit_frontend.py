import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from rag_utils import start_ingestion_async, get_ingestion_status
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command
from rag_utils import get_retriever_for_thread
import uuid
import os
import time

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return str(content)

def generate_thread_id():
    return uuid.uuid4()

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    pending = None
    if state.tasks:
        for task in state.tasks:
            if task.interrupts:
                pending = task.interrupts[0].value
                break

    return messages, pending

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])
st.sidebar.title("RAG Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.divider()
st.sidebar.subheader("Upload a PDF")

uploaded_pdf = st.sidebar.file_uploader("Choose a PDF for this chat", type=["pdf"])
current_thread_id = str(st.session_state["thread_id"])
if get_retriever_for_thread(current_thread_id) is not None:
    st.sidebar.success("Document loaded for this chat")
else:
    st.sidebar.info("No document loaded yet")

if uploaded_pdf is not None:
    if st.sidebar.button("Process PDF"):
        temp_path = f"temp_{uploaded_pdf.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.getvalue())

        start_ingestion_async(
            temp_path,
            thread_id=str(st.session_state["thread_id"]),
            filename=uploaded_pdf.name
        )
        st.session_state["ingesting_temp_path"] = temp_path
        st.rerun()

# Poll for in-progress ingestion on every rerun (this is what makes it "async"
# from the UI's perspective — Streamlit reruns the whole script on a timer
# while we're polling, so the user sees live progress instead of one frozen
# blocking spinner).
if "ingesting_temp_path" in st.session_state:
    status = get_ingestion_status(current_thread_id)

    if status is None:
        pass  # thread hasn't posted a status yet, will show up next rerun
    elif status["status"] == "running":
        st.sidebar.info(f"⏳ {status['stage']}")
        time.sleep(0.5)
        st.rerun()
    elif status["status"] == "done":
        st.sidebar.success(f"Indexed {uploaded_pdf.name if uploaded_pdf else ''} ({status['result']['chunks']} chunks)")
        try:
            os.remove(st.session_state["ingesting_temp_path"])
        except OSError:
            pass
        del st.session_state["ingesting_temp_path"]
    elif status["status"] == "error":
        st.sidebar.error(f"Ingestion failed: {status['error']}")
        try:
            os.remove(st.session_state["ingesting_temp_path"])
        except OSError:
            pass
        del st.session_state["ingesting_temp_path"]

st.sidebar.divider()

st.sidebar.header("My Conversations")
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(extract_text(message["content"]))

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages, pending = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            if extract_text(msg.content).strip():
                temp_messages.append({"role": role, "content": msg.content})

        st.session_state["message_history"] = temp_messages
        st.session_state["pending_interrupt"] = pending
        st.rerun()

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": str(st.session_state["thread_id"])},
    "run_name": "streamlit_chat_turn",
}

def handle_graph_result(result):
    """Store the final answer or a pending interrupt, based on what the graph returned."""
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        st.session_state["pending_interrupt"] = payload
    else:
        st.session_state["pending_interrupt"] = None
        answer = result["messages"][-1].content
        st.session_state["message_history"].append(
            {"role": "assistant", "content": extract_text(answer)}
        )


if st.session_state["pending_interrupt"] is not None:
    payload = st.session_state["pending_interrupt"]
    st.warning(
        f"**Approval needed** — tool: `{payload['tool_name']}`\n\nArgs: `{payload['tool_args']}`"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve"):
            result = chatbot.invoke(Command(resume="approve"), config=CONFIG)
            handle_graph_result(result)
            st.rerun()
    with col2:
        if st.button("❌ Reject"):
            result = chatbot.invoke(Command(resume="reject"), config=CONFIG)
            handle_graph_result(result)
            st.rerun()

else:
    user_input = st.chat_input("Type here")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG
        )
        handle_graph_result(result)
        st.rerun()