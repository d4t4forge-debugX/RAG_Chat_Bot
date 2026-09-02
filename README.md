# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions from an uploaded PDF, with hybrid search, human-in-the-loop approval for web search, conversation persistence, and RAGAS-based evaluation. Built as a placement-interview portfolio project — every design choice below is intentional and documented, not accidental.

<!-- Screenshot / GIF placeholder — add a screenshot of the chat UI here once available -->
<!-- ![App screenshot](docs/screenshot.png) -->

**Live demo:** coming soon (deployment is the final step of this project — see [Roadmap](#roadmap))

---

## What it does

Upload a PDF, ask questions about it, and get answers grounded in the document's actual content. If the document doesn't cover something, the assistant can search the web instead — but only after you explicitly approve that specific search, so nothing leaves your machine without permission. The system is also self-evaluating: it ships with a RAGAS-based evaluation harness that scores answer faithfulness, relevancy, and context precision, so retrieval quality isn't just assumed — it's measured.

## Key features

- **Hybrid retrieval (FAISS + BM25)** — combines dense semantic search with sparse keyword search via LangChain's `EnsembleRetriever`, tuned to reduce keyword-search noise (see [Design Decisions](#key-design-decisions--trade-offs))
- **Human-in-the-loop approval for web search** — every web search is paused via LangGraph's `interrupt()` and requires explicit user approval before it runs
- **Streamed responses** — answers stream token-by-token in the UI instead of appearing all at once, using LangGraph's `stream()` API filtered to the correct graph node
- **Conversation persistence** — each chat thread is saved via `SqliteSaver`, so conversations (including paused approvals) survive page reloads and thread switching
- **Async PDF ingestion** — PDF processing runs on a background thread with live progress feedback in the UI, so the interface never freezes during embedding
- **In-memory query caching** — repeated questions within a session skip re-retrieval entirely (measured ~150x speedup on cache hits)
- **Swappable LLM backend** — the model is selected through one config function, overridable via environment variable, with no other file needing to change
- **Input guardrails** — a lightweight LLM classifier screens out inappropriate queries before they reach the main chat flow
- **Automated tests** — 15 pytest tests covering tool logic, guardrail parsing, and caching behavior, independent of live API calls
- **RAGAS evaluation harness** — measures faithfulness, answer relevancy, and context precision against a fixed question set, so retrieval changes can be validated with numbers, not vibes

## Architecture

```
START
  │
  ▼
guardrail_node  ──(inappropriate query)──▶ END (refusal message)
  │
  ▼ (query passes)
chat_node ◀────────────────────────────────┐
  │                                         │
  ├─(no tool needed)──▶ END                 │
  │                                         │
  ├─(wants web search)─▶ human_review_node  │
  │                         │               │
  │                         ├─approved──▶ tools ─┘
  │                         └─rejected──▶ (back to chat_node with denial message)
  │                                         │
  └─(wants calculator / doc search)─▶ tools ┘
```

The graph is built with **LangGraph**. `chat_node` is the only node that calls the LLM for a real response; `guardrail_node` runs a separate, cheap classification call before it. Only `duckduckgo_search` is routed through `human_review_node` — the calculator and document-retrieval tools are deterministic and internal, so they run without approval.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) | Explicit state machine with native support for human-in-the-loop interrupts |
| LLM | Google Gemini (free tier) | No per-call cost during development |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, runs offline, no API dependency for retrieval |
| Vector store | FAISS | Fast local similarity search, no external service required |
| Lexical search | BM25 (`rank_bm25`) | Catches exact keyword matches dense embeddings can miss |
| Persistence | SQLite via `SqliteSaver` | Zero-setup conversation and checkpoint storage |
| Web search | DuckDuckGo (`ddgs`) | No API key required |
| Evaluation | [RAGAS](https://github.com/explodinggradients/ragas) | Standard, LLM-graded RAG evaluation metrics |
| Observability | LangSmith | Trace-level visibility into every graph run |
| Frontend | Streamlit | Fast to build, good fit for a demo-first project |
| Testing | pytest | Standard, widely recognized |

## Key design decisions & trade-offs

These are the choices I expect to be asked about in an interview, along with the reasoning behind each one.

**Hybrid retrieval tuning — reweighting alone doesn't work.**
`EnsembleRetriever` merges the *union* of results from each retriever and only re-ranks by weight; it doesn't drop low-weight candidates. I initially assumed changing `weights=[0.5,0.5]` to `weights=[0.8,0.2]` would filter out noisy BM25 matches — it didn't, since the same candidate set surfaced under both configurations. The BM25 sub-retriever was surfacing an entire "Exercises" section of the source PDF as false-positive keyword matches. The actual fix was reducing BM25's own `k` (`bm25_k=2`) at the retrieval source, not adjusting the ensemble weights. This is documented because it's a good example of verifying a fix against real output rather than assuming a plausible-sounding change worked.

**Human-in-the-loop gates only web search, not every tool.**
Only `duckduckgo_search` requires explicit approval. The calculator and document-retrieval tool are deterministic, don't touch external/untrusted sources, and gating them would just add friction with no safety benefit. HITL is applied where it actually matters: the one tool making an uncontrolled external network call.

**Guardrails are prompt-based, not a separate trained classifier.**
Input filtering is a single cheap LLM call that returns YES/NO before the main chat flow runs; output groundedness (not fabricating facts beyond retrieved context) is enforced via system-prompt instructions rather than a post-hoc verification model. This is simpler to build, explain, and reason about than a dedicated classifier, at the cost of being less robust than a purpose-trained model — an accepted trade-off for a project at this scope.

**Caching covers retrieval, not final LLM output.**
`rag_tool`'s cache stores retrieved chunks keyed by `(thread_id, query)`, not the LLM's generated answer. Caching final answers raises staleness questions (e.g., if the system prompt changes, a cached answer could reflect old instructions) that outweigh the benefit at this scale — so caching was deliberately scoped to the deterministic part of the pipeline only.

**Single document per conversation thread, with an explicit replace confirmation.**
Uploading a second PDF to a thread that already has one loaded doesn't silently overwrite it — the UI shows an explicit "this will replace your current document" warning requiring a second confirmation click. Supporting multiple simultaneous documents per thread was considered and deliberately deferred: it adds real complexity (merged retrieval across documents, cache-key changes, UI for showing which documents are active) that isn't proportional to its interview value for this project's scope. One document per conversation is a clean, easy-to-defend boundary; the confirmation step exists specifically so that boundary doesn't look like a bug during a live demo.

**Streaming covers new chat turns; the approval-resume path does not.**
Token streaming was added for normal conversation turns via `chatbot.stream()`, but resuming after an approve/reject decision still uses a blocking call. Resumed responses are typically short, and this keeps the two code paths independently simple rather than merging streaming logic into the interrupt-resume flow for marginal benefit.

**Local embeddings over a paid embeddings API.**
`sentence-transformers/all-MiniLM-L6-v2` runs locally with no per-call cost and no external dependency for the retrieval step — a deliberate budget-conscious choice that also removes one more network dependency from the critical path.

## Project structure

```
RAG_Chat_Bot/
├── langgraph_backend.py     # Graph definition: nodes, routing, tools, guardrails
├── streamlit_frontend.py    # Streamlit UI: chat, PDF upload, approval flow, streaming
├── rag_utils.py              # PDF ingestion, hybrid retriever, rag_tool, caching
├── llm_config.py              # Single source of truth for LLM selection
├── evaluation.py               # RAGAS evaluation harness
├── test_calculator.py          # Unit tests: calculator tool
├── test_guardrail.py            # Unit tests: guardrail YES/NO parsing (mocked LLM)
├── test_rag_caching.py           # Unit tests: rag_tool caching behavior (mocked retriever)
├── requirements.txt
└── .gitignore
```

## Setup

**Requirements:** Python 3.11+ (developed on 3.14), a free [Google AI Studio](https://aistudio.google.com/) API key.

```bash
# Clone the repo
git clone https://github.com/d4t4forge-debugX/RAG_Chat_Bot.git
cd RAG_Chat_Bot

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

(Optional) To use a different Gemini model without touching any code:

```
LLM_MODEL=gemini-2.5-flash
LLM_THINKING_LEVEL=low
```

## Running it

```bash
streamlit run streamlit_frontend.py
```

Open the local URL Streamlit prints in your terminal, upload a PDF from the sidebar, and start asking questions.

## Running the tests

```bash
pytest -v
```

15 tests covering the calculator tool, guardrail response parsing, and `rag_tool`'s caching logic — all run without hitting any real LLM or retriever, using mocks where an external call would otherwise be required.

## Running the evaluation

```bash
python evaluation.py
```

Runs a fixed set of test questions against a freshly-ingested PDF and scores the results with RAGAS. Current baseline (hybrid retrieval, `bm25_k=2`, `weights=[0.7, 0.3]`):

| Metric | Score |
|---|---|
| Faithfulness | 0.83 – 0.93 (varies run-to-run; see note below) |
| Answer relevancy | 0.81 – 0.88 |
| Context precision | 0.53 (stable across runs) |

Faithfulness and answer relevancy vary slightly between runs — this was investigated and traced to RAGAS's own LLM-judge noise on a small (n=3) evaluation set, confirmed by re-running the evaluation independently and observing that context precision (which doesn't depend on judge subjectivity) stayed identical both times.

## Known limitations (deliberately out of scope)

Documented explicitly rather than left implicit, since a project with zero acknowledged limitations is less credible than one with clearly reasoned boundaries:

- **No multi-user support / no authentication** — this is a single-user local/demo project, not a production multi-tenant system.
- **In-memory per-thread state resets on restart** — retrievers and query caches are not persisted to disk; a redeploy clears them. Acceptable for a demo project; a production version would use a persistent vector store.
- **One document per conversation thread** — see [Design Decisions](#key-design-decisions--trade-offs) above.
- **No semantic chunking** — uses fixed-size recursive character splitting rather than embedding-aware chunk boundaries; simpler and sufficient for this scope.
- **No cost/quota-aware model routing** — the LLM is fixed per environment variable, not dynamically switched based on usage or cost.

## Roadmap

- [ ] README (this file)
- [ ] Resume/portfolio writeup
- [ ] Deploy to Streamlit Community Cloud
- [ ] Re-verify all features against the live deployed app

## License

MIT — see [LICENSE](LICENSE). Chosen for simplicity and because it places no restrictions on reuse, which fits a public portfolio project.
