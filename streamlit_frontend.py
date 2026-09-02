import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from rag_utils import start_ingestion_async, get_ingestion_status, get_retriever_for_thread
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command
import uuid
import os
import time


# pulls plain display text out of a complete (non-streamed) message's content, handling both str and block-list formats
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


# pulls visible text out of one streamed message chunk, concatenating with no separator so words don't get split
def extract_stream_chunk_text(content):
    """
    Extract just the visible text delta from a single streamed message
    chunk. Unlike extract_text() (used on a final, complete message),
    streaming chunks are small partial fragments that must be concatenated
    with no separator, or words get broken up with stray spaces. Chunks
    that are internal 'thinking'/signature metadata (no 'text' type block)
    contribute nothing.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


# creates a brand-new random thread_id for a new conversation
def generate_thread_id():
    return uuid.uuid4()


# registers a thread_id in the sidebar's thread list if it isn't already tracked
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# sweeps temp PDF files for any thread whose background ingestion has finished (done or error), regardless of active thread
def cleanup_finished_ingestions():
    """
    Checks every thread we're tracking a temp file for. If that thread's
    ingestion has finished (done or error), delete its temp file and stop
    tracking it. Still-running ingestions are left alone, even if the user
    has navigated to a different thread — this is what lets background
    ingestion survive switching threads or starting a new chat.
    """
    finished = []
    for tid, temp_path in st.session_state["ingesting_temp_paths"].items():
        status = get_ingestion_status(tid)
        if status is not None and status["status"] in ("done", "error"):
            try:
                os.remove(temp_path)
            except OSError:
                pass
            finished.append(tid)

    for tid in finished:
        del st.session_state["ingesting_temp_paths"][tid]

# starts a fresh conversation: clears history, pending interrupt, and generates a new thread_id
def reset_chat():
    cleanup_finished_ingestions()
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None


# reloads a saved thread's message history from the checkpointer, plus any pending HITL interrupt still paused on it
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


# ============================ Session Setup ============================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

if "ingesting_temp_paths" not in st.session_state:
    st.session_state["ingesting_temp_paths"] = {}

if "confirm_pdf_replace" not in st.session_state:
    st.session_state["confirm_pdf_replace"] = False

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])

# ============================== Sidebar =================================

st.sidebar.title("RAG Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.divider()
st.sidebar.subheader("Upload a PDF")

uploaded_pdf = st.sidebar.file_uploader("Choose a PDF for this chat", type=["pdf"])
current_thread_id = str(st.session_state["thread_id"])

# writes the uploaded PDF to a thread-scoped temp path and kicks off background ingestion for it
def _start_pdf_ingestion(pdf_file):
    """Shared by both the normal and confirmed-replace paths below, so the
    actual ingestion kickoff logic lives in exactly one place."""
    temp_path = f"temp_{current_thread_id}_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())

    start_ingestion_async(
        temp_path,
        thread_id=current_thread_id,
        filename=pdf_file.name
    )
    st.session_state["ingesting_temp_paths"][current_thread_id] = temp_path
    st.session_state["confirm_pdf_replace"] = False
    st.rerun()


if uploaded_pdf is not None:
    existing_doc_loaded = get_retriever_for_thread(current_thread_id) is not None

    if not existing_doc_loaded:
        # Nothing to replace — proceed exactly as before, no confirmation needed.
        if st.sidebar.button("Process PDF"):
            _start_pdf_ingestion(uploaded_pdf)

    elif not st.session_state["confirm_pdf_replace"]:
        # A document is already loaded for this thread. First click just
        # asks for confirmation instead of immediately overwriting it.
        if st.sidebar.button("Process PDF"):
            st.session_state["confirm_pdf_replace"] = True
            st.rerun()

    else:
        # User already clicked "Process PDF" once with a doc loaded — now
        # show the explicit warning and require a second, distinct click.
        st.sidebar.warning("This will replace the document currently loaded for this chat.")
        if st.sidebar.button("Yes, replace it"):
            _start_pdf_ingestion(uploaded_pdf)
        if st.sidebar.button("Cancel"):
            st.session_state["confirm_pdf_replace"] = False
            st.rerun()

# Check ingestion status directly by thread_id, every rerun — regardless of
# whether the user navigated away and back. This is the core fix: status
# lives in rag_utils' _INGESTION_STATUS dict (keyed by thread_id), not in
# fragile per-session UI state, so it displays correctly no matter how the
# user got to this thread.
current_status = get_ingestion_status(current_thread_id)

if current_status is not None and current_status["status"] == "running":
    st.sidebar.info(f"⏳ {current_status['stage']}")
    time.sleep(0.5)
    st.rerun()
elif current_status is not None and current_status["status"] == "error":
    st.sidebar.error(f"Ingestion failed: {current_status['error']}")
elif current_status is not None and current_status["status"] == "done":
    result = current_status["result"]
    st.sidebar.success(f"Indexed {result['filename']} ({result['chunks']} chunks)")
elif get_retriever_for_thread(current_thread_id) is not None:
    st.sidebar.success("Document loaded for this chat")
else:
    st.sidebar.info("No document loaded yet")

# Sweep up temp files for any thread whose ingestion has actually finished,
# not just the one we're currently looking at.
cleanup_finished_ingestions()

# Poll for in-progress ingestion on every rerun (this is what makes it "async"
# from the UI's perspective — Streamlit reruns the whole script on a timer
# while we're polling, so the user sees live progress instead of one frozen
# blocking spinner).
# NOTE: this whole block checks a session_state key ("ingesting_temp_path", singular)
# that is never set anywhere in this file anymore — it is dead/unreachable code left
# over from the old single-thread tracking design; flagged above, not removed here
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

# thread-switch loop: loads the selected thread's history + any pending interrupt, then reruns to refresh the view
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        cleanup_finished_ingestions()
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

# ============================ Main Chat Area =============================

# renders every stored message in the current thread's history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(extract_text(message["content"]))

CONFIG = {
    "configurable": {"thread_id": st.session_state["thread_id"]},
    "metadata": {"thread_id": str(st.session_state["thread_id"])},
    "run_name": "streamlit_chat_turn",
}


# used after an approve/reject .invoke() call: stores either a new pending interrupt or the final answer in history
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

# generator for st.write_stream(): streams only chat_node's assistant tokens and accumulates the full text as it goes
def stream_chat_turn(user_input, config, accumulated_holder):
    """
    Generator fed into st.write_stream(). Streams chat_node's text and
    guardrail_node's refusal message, while explicitly excluding
    guardrail_node's internal YES/NO classifier call via its
    "guardrail_classifier" tag — tag-based, not text-based, so a real
    answer that happens to start with "No" is never mistakenly filtered.
    """
    for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
    ):
        if metadata.get("langgraph_node") not in ("chat_node", "guardrail_node"):
            continue
        if "guardrail_classifier" in metadata.get("tags", []):
            continue
        text_piece = extract_stream_chunk_text(message_chunk.content)
        if text_piece:
            accumulated_holder[0] += text_piece
            yield text_piece

# top-level branch: if a tool call is awaiting approval, show the approve/reject card instead of the chat input
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
    # normal turn: take chat input, stream the assistant's reply live, then check whether it paused on an interrupt
    user_input = st.chat_input("Type here")

    if user_input:
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.text(user_input)

        accumulated_holder = [""]
        with st.chat_message("assistant"):
            st.write_stream(stream_chat_turn(user_input, CONFIG, accumulated_holder))

        state = chatbot.get_state(config=CONFIG)
        pending = None
        if state.tasks:
            for task in state.tasks:
                if task.interrupts:
                    pending = task.interrupts[0].value
                    break

        if pending is not None:
            st.session_state["pending_interrupt"] = pending
        else:
            st.session_state["pending_interrupt"] = None
            st.session_state["message_history"].append(
                {"role": "assistant", "content": accumulated_holder[0]}
            )

        st.rerun()