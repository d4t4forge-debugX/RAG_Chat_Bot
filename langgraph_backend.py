from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from rag_utils import rag_tool, ingest_pdf_for_thread
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langgraph.types import interrupt, Command
load_dotenv()

# ---- Tools defined FIRST ----
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}
def is_query_appropriate(query: str) -> bool:
    """
    Lightweight check: rejects clearly off-topic or unsafe queries
    before they reach the main chat flow.
    """
    check_prompt = (
        "You are a content filter. Answer with only 'YES' or 'NO'.\n"
        "Is the following user message a reasonable question for a helpful "
        "document/research assistant to answer? Answer 'NO' only if it is "
        "clearly abusive, asks for illegal content, or is obviously spam.\n\n"
        f"Message: {query}"
    )
    response = llm.invoke(check_prompt)
    answer = response.content
    if isinstance(answer, list):
        answer = " ".join(
            block.get("text", "") for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return "yes" in answer.lower()

search_tool = DuckDuckGoSearchRun(region="us-en")

# ---- LLM setup AFTER tools exist ----
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", thinking_level="low")

tools = [search_tool, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def guardrail_node(state: ChatState):
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, HumanMessage):
        query_text = last_message.content
        if isinstance(query_text, list):
            query_text = " ".join(
                block.get("text", "") for block in query_text
                if isinstance(block, dict) and block.get("type") == "text"
            )

        if not is_query_appropriate(query_text):
            refusal = AIMessage(
                content="I'm not able to help with that request. Please ask a different question."
            )
            return {"messages": [refusal]}

    return {"messages": []}


def chat_node(state: ChatState, config=None):
    messages = state["messages"]

    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a research assistant. If the user asks about their uploaded document, "
            f"call the rag_tool and pass thread_id='{thread_id}'. "
            "You can also use the calculator and web search tools when helpful.\n\n"
            "For complex or multi-part questions, break them into smaller sub-questions "
            "and call rag_tool multiple times with different, specific queries if needed, "
            "rather than relying on a single search. Only give your final answer once you "
            "have gathered enough information to answer completely and accurately. "
            "If a search doesn't return useful information, try rephrasing the query and "
            "search again before giving up.\n\n"
            "IMPORTANT: Only make claims that are directly supported by the retrieved "
            "document content or tool results. If the retrieved information doesn't "
            "contain a clear answer to the question, say so explicitly rather than "
            "guessing or relying on general knowledge. For example, say: "
            "'The uploaded document doesn't appear to cover this topic.' "
            "Never fabricate facts, page numbers, or details that aren't in the retrieved context.\n\n"
            "Do not name specific methods, techniques, formulas, or related concepts "
            "(e.g., alternative algorithms, named equations, or related terminology) "
            "unless they explicitly appear in the retrieved context, even if you know "
            "from general knowledge that they are relevant. If you're tempted to add a "
            "related term or example that isn't in the retrieved text, leave it out."
        )
    )

    full_messages = [system_message, *messages]
    response = llm_with_tools.invoke(full_messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)
def human_review_node(state: ChatState):
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[0]

    decision = interrupt({
        "question": "Approve this tool call?",
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
    })

    if decision == "approve":
        return Command(goto="tools")
    else:
        denial = ToolMessage(
            content=(
                "The user explicitly declined to approve this tool call. "
                "Do not retry this tool or attempt a similar search. "
                "Instead, tell the user the search was not approved and ask if they'd like to proceed differently."
            ),
            tool_call_id=tool_call["id"],
        )
        return Command(goto="chat_node", update={"messages": [denial]})

def route_after_chat(state: ChatState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return END
    if last_message.tool_calls[0]["name"] == "duckduckgo_search":
        return "human_review_node"
    return "tools"

def route_after_guardrail(state: ChatState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and "not able to help" in str(last_message.content):
        return END
    return "chat_node"


graph = StateGraph(ChatState)
graph.add_node("guardrail_node", guardrail_node)
graph.add_node("chat_node", chat_node)
graph.add_node("human_review_node", human_review_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "guardrail_node")
graph.add_conditional_edges("guardrail_node", route_after_guardrail)
graph.add_conditional_edges("chat_node", route_after_chat)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

if __name__ == "__main__":
    import uuid
    test_thread_id = f"hitl-test-{uuid.uuid4()}"
    CONFIG = {
        "configurable": {"thread_id": test_thread_id},
        "metadata": {"thread_id": test_thread_id},
        "run_name": "hitl_smoke_test",
    }

    print("Asking a question that should trigger a web search...\n")
    result = chatbot.invoke(
        {"messages": [HumanMessage(content="Search the web for the latest LangGraph release version.")]},
        config=CONFIG
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("PAUSED FOR APPROVAL:")
        print(f"  Tool: {payload['tool_name']}")
        print(f"  Args: {payload['tool_args']}")

        decision = input("\nApprove this tool call? (approve/reject): ").strip().lower()
        result = chatbot.invoke(Command(resume=decision), config=CONFIG)

    answer = result["messages"][-1].content
    if isinstance(answer, list):
        answer = " ".join(
            block.get("text", "") for block in answer
            if isinstance(block, dict) and block.get("type") == "text"
        )
    print("\nFinal answer:")
    print(answer)