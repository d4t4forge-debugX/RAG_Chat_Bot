# RAG Chatbot — Project Notes

Project path: `/Users/hawkeyez007/Desktop/RAG_Chat_Bot`
Stack: Python 3.14, LangGraph, Streamlit, Gemini (`gemini-3.5-flash-lite`, free tier),
local HuggingFace embeddings, FAISS, SQLite persistence, DuckDuckGo search (`ddgs`), RAGAS eval.

## ⚠️ Manual patch required after any fresh venv rebuild

`ragas==0.3.9` has an upstream bug: it unconditionally imports
`langchain_community.chat_models.vertexai`, a module path that no longer
exists in current `langchain-community` versions. This breaks any script
that imports `ragas` (i.e. `evaluation.py`) with:

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

**This is a manual edit to an installed package — `requirements.txt` cannot
capture it. It must be reapplied after every fresh `pip install`.**

File to patch: `.venv/lib/python3.14/site-packages/ragas/llms/base.py`

Fix: wrap the `ChatVertexAI` / `VertexAI` imports in a try/except, falling
back to `langchain_google_vertexai` (already a pinned dependency), and
filter `None` out of the `MULTIPLE_COMPLETION_SUPPORTED` list so downstream
`isinstance()` checks don't break on a `None` entry.

After patching, also clear ragas's `__pycache__` directory — a stale
compiled `.pyc` can mask the fix even after the source is corrected:

    find .venv -path "*/ragas/**/__pycache__" -exec rm -rf {} +