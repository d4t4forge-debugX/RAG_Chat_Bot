# RAG Chatbot — Project Notes

Project path: `/Users/hawkeyez007/Desktop/RAG_Chat_Bot`
Goal: Resume-ready RAG chatbot for placement interviews. Python 3.14, PyCharm,
Streamlit frontend, LangGraph backend, Gemini free tier, local embeddings.

## Stack
- LLM: Google Gemini `gemini-3.5-flash-lite` (`thinking_level="low"`)
- Embeddings: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS
- Lexical retrieval: BM25 (`rank_bm25` package, via `langchain_community.retrievers.BM25Retriever`)
- Hybrid retrieval: `EnsembleRetriever` (Reciprocal Rank Fusion) — **now lives in
  `langchain_classic.retrievers`, not `langchain.retrievers`**, as of LangChain v1.0
- Persistence: `SqliteSaver` (LangGraph), db file `chatbot.db`
- Web search: `ddgs` package (renamed from `duckduckgo-search`)
- Evaluation: RAGAS 0.3.9

## Files
- `langgraph_backend.py` — LangGraph graph: guardrail_node → chat_node ⇄ tools,
  with human_review_node gating web search. Checkpointer, retrieve_all_threads().
- `streamlit_frontend.py` — Streamlit UI: sidebar threads, PDF upload, HITL
  approve/reject cards, thread-switch with pending-interrupt restoration.
- `rag_utils.py` — PDF loading/chunking/cleaning, embeddings, FAISS, BM25, hybrid
  ensemble retriever, per-thread retriever store (`_THREAD_RETRIEVERS`,
  `_THREAD_METADATA` dicts), `ingest_pdf_for_thread()`, `get_retriever_for_thread()`,
  `rag_tool`.
- `evaluation.py` — standalone RAGAS harness: ingests test.pdf fresh, runs 3
  test questions through the real chatbot, scores with RAGAS, saves
  `evaluation_results.json`.
- `requirements.txt` — curated direct dependencies (not a raw pip freeze).
- `test.pdf` — Hands-On Machine Learning by Aurélien Géron (58MB, gitignored —
  not committed to the repo; copyrighted, redistribute-your-own-copy needed).

## Architecture (current graph shape)

START → guardrail_node → (conditional: route_after_guardrail)
├─ inappropriate → END (refusal message)
└─ appropriate → chat_node → (conditional: route_after_chat)
├─ duckduckgo_search → human_review_node
│ → approve → tools → chat_node (loop)
│ → reject → chat_node (denial ToolMessage)
├─ other tool → tools → chat_node
└─ no tool → END

- `ChatState`: `TypedDict` with `messages: Annotated[list[BaseMessage], add_messages]`
- Tools bound to LLM: `search_tool` (DuckDuckGo), `calculator`, `rag_tool`
- Only `search_tool` (external/untrusted network call) is gated behind human
  approval via `interrupt()`/`Command` — not `calculator` or `rag_tool`
  (deterministic/internal, no approval needed)

## Retrieval architecture (hybrid search — in progress)

`get_retriever(vector_store, chunks, k=4, bm25_k=2)` in `rag_utils.py` builds a
hybrid retriever:
- FAISS (semantic/dense) retriever: returns top `k=4` by vector similarity
- BM25 (lexical/keyword) retriever: returns top `bm25_k=2` by term frequency
- Combined via `EnsembleRetriever(retrievers=[faiss, bm25], weights=[0.7, 0.3])`
  using weighted Reciprocal Rank Fusion

**Important mechanical detail**: `EnsembleRetriever` merges the **union** of each
sub-retriever's results and only *re-ranks* via weighting — it does NOT drop
candidates based on weight. The only way to reduce a sub-retriever's noise
contribution is to lower its own `k`, not just its weight. (Confirmed by testing:
identical result *sets* appeared under `weights=[0.5,0.5]` and `weights=[0.8,0.2]`
with equal per-retriever `k` — only the ranking order changed.)

**Why hybrid at all**: FAISS/semantic-only retrieval can miss exact keyword/
lexical matches (acronyms, rare terms, specific numbers). BM25 adds classic
term-frequency matching to catch those cases. Combining tends to outperform
either alone in production RAG systems.

**Known issue found via RAGAS + manual chunk inspection (not yet fully
resolved)**: this specific PDF (Hands-On ML textbook) has many end-of-chapter
"Exercises" sections — short, numbered, keyword-dense questions covering many
ML terms in a small space. BM25 ranks these highly for almost any ML query
even though they're not explanatory content, because of raw keyword density.
FAISS mostly avoids them but its own 3rd/4th-ranked picks aren't always
strong either — the corpus itself is noisy for broad queries, independent of
BM25.

**Debugging trail**:
1. Built hybrid retriever with `weights=[0.5, 0.5]`, `k=4` for both retrievers →
   RAGAS scores dropped across the board vs. pre-hybrid baseline:
   - faithfulness: 0.9583 → 0.8561
   - answer_relevancy: 0.8203 → 0.7598
   - context_precision: 0.5111 → 0.3921
2. Manually inspected `evaluation_results.json` contexts — confirmed exercise-
   list chunks were making it into the LLM's context for all 3 test questions
   (roughly 4 of 8 retrieved chunks were off-topic exercise lists).
3. Tried reweighting to `[0.8, 0.2]` — **result set was unchanged**, only
   reordered. This revealed the EnsembleRetriever union/rerank mechanic above.
4. Correct fix in progress: added independent `bm25_k` parameter (default 2)
   to shrink BM25's candidate pool at the source rather than relying on
   weighting. Manually verified this reduces exercise-list chunks in raw
   retrieval output (from 4/8 to 2/6 for the "What is supervised learning?"
   query).
5. **Not yet done**: re-run `evaluation.py` with `bm25_k=2, weights=[0.7,0.3]`
   to get updated RAGAS numbers and confirm whether this recovers toward the
   pre-hybrid baseline. This is the next immediate step.

## Key Decisions & Why
- **Local embeddings over OpenAI**: no per-call cost, good interview talking point
- **Gemini free tier**: budget constraint drove model choice and `thinking_level="low"`
- **RAG state in-memory, keyed by thread_id**: per-conversation PDF isolation;
  known limitation — resets on app restart, documented as future work
- **HITL gates only web search, not calculator/rag_tool**: only tool making
  external/untrusted network calls; interview-defensible reasoning
- **Guardrails are prompt-based, not separate classifiers**: input guardrail
  (`is_query_appropriate`) is one cheap permissive LLM call short-circuiting
  to END; output groundedness is system-prompt instructions, not a post-hoc
  verification step — simpler to build/explain, standard technique
- **k=6 retrieval (pre-hybrid work, changed from k=4)**: increases context
  available to answer without ballooning prompt size too much on free-tier.
  Note: hybrid search work reintroduced `k=4` as the FAISS default while
  tuning — reconcile this with the k=6 decision once hybrid tuning is finalized.
- **Hybrid search added as a stretch feature** (Days 20 roadmap explicitly
  deferred this) to demonstrate retrieval depth beyond pure vector search —
  chosen over LangSmith tracing / LLM-swap abstraction as the first stretch
  feature to build.
- **requirements.txt hand-curated, not `pip freeze`**: direct dependencies only,
  mostly unpinned (except ragas==0.3.9, which must stay pinned for the manual
  patch below to apply)

## Bugs Fixed
1. **`message`/`messages` key typo** in early `ChatState` — traced via traceback
2. **Tools referenced before definition** — Python top-to-bottom execution order
3. **`duckduckgo-search` → `ddgs` package rename** — updated import + requirements
4. **Broken Unicode surrogates breaking embedding tokenizer** — PDF math notation
   produced orphaned surrogate chars; fixed with `re.sub(r'[\ud800-\udfff]', '', text)`
   in `rag_utils.py`'s `clean_text()`, applied during PDF loading
5. **Gemini model deprecations** — `gemini-2.5-flash` retired, `gemini-3.6-flash-lite`
   never existed; settled on `gemini-3.5-flash-lite`
6. **Free-tier quota exhaustion (429)** on `gemini-3.6-flash` — switched to
   `gemini-3.5-flash-lite` (separate/higher quota bucket)
7. **`ragas` import crash** — see "Manual patch required" section below
8. **HITL: LLM retried rejected tool calls** — treated "rejected" same as "search
   failed" and retried with a new query, causing uncaught second interrupt.
   Fixed by making the denial `ToolMessage` explicit ("do not retry this tool"),
   and making test/UI code loop over interrupts rather than assume exactly one
9. **Streamlit thread-switch staleness** — clicking a sidebar thread button
   updated session_state but didn't `st.rerun()`. Fixed by adding `st.rerun()`
   right after the state updates in the thread-switch loop.
10. **Debug print flooding terminal** — removed a stray debug print from
    `handle_graph_result` in streamlit_frontend.py
11. **Leftover temp PDF files never deleted** — upload handler wrote
    `temp_<filename>.pdf` and never cleaned up; added `os.remove()` after
    ingestion completes
12. **Duplicate dead code** in `langgraph_backend.py`'s `__main__` block —
    final-answer print logic was pasted twice; removed the duplicate
13. **Leaked API key caught before real damage** — key rotated, `.env`
    untracked via `git rm --cached -f`
14. **`.idea/` and 58MB `test.pdf` committed despite .gitignore** — removed
    via `git rm --cached` after the fact
15. **RAGAS faithfulness anomaly (0.2674) investigated and resolved** (pre-
    hybrid-search session) — root cause was the model adding claims not in
    retrieved context (naming Ridge/Elastic Net when only Lasso was retrieved).
    Fixed via k=4→6 retrieval + tightened system prompt forbidding naming
    unretrieved methods/concepts. Also fixed a separate RAGAS/Gemini
    incompatibility (`gemini-3.5-flash-lite` doesn't support multi-candidate
    generation) via `AnswerRelevancy(strictness=1)`.
16. **`ingest_pdf_for_thread` broke after `get_retriever` signature change**
    (hybrid search work) — `get_retriever` gained a required `chunks` param
    for BM25, but `ingest_pdf_for_thread` still called it with only
    `vector_store`. Fixed by updating the call site. Caught before running,
    by manually cross-checking all call sites against the new signature.
17. **`ModuleNotFoundError: langchain.retrievers`** (hybrid search work) —
    LangChain v1.0 moved `EnsembleRetriever` out of the core `langchain`
    package into a new separate `langchain-classic` package. Fixed via
    `pip install langchain-classic` + import path change to
    `langchain_classic.retrievers`. `BM25Retriever` was unaffected — stayed
    in `langchain_community.retrievers`.

## Manual patch required after any fresh venv rebuild
`ragas==0.3.9` has an upstream bug: it unconditionally imports
`langchain_community.chat_models.vertexai`, a module path that no longer
exists in current `langchain-community` versions. Breaks any script
importing `ragas` (i.e. `evaluation.py`) with:
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

**This is a manual edit to an installed package — requirements.txt cannot
capture it. Must be reapplied after every fresh `pip install`.**

File to patch: `.venv/lib/python3.14/site-packages/ragas/llms/base.py`
Fix: wrap `ChatVertexAI`/`VertexAI` imports in try/except, falling back to
`langchain_google_vertexai` (already a pinned dependency), and filter `None`
out of `MULTIPLE_COMPLETION_SUPPORTED` so downstream `isinstance()` checks
don't break.

After patching, clear ragas's `__pycache__` — a stale `.pyc` can mask the fix:
    find .venv -path "*/ragas/**/__pycache__" -exec rm -rf {} +

## RAGAS Baselines (for comparison across experiments)

**Pre-hybrid-search baseline** (k=6, FAISS-only, tightened prompt, strictness=1):
    faithfulness: 0.9583
    answer_relevancy: 0.8203
    context_precision: 0.5111
This is the last known-good trustworthy baseline before hybrid search work began.

**Hybrid search, `weights=[0.5,0.5]`, `k=4` both retrievers** (regressed —
root-caused to BM25 noise, see "Retrieval architecture" section above):
    faithfulness: 0.8561
    answer_relevancy: 0.7598
    context_precision: 0.3921

**Hybrid search, `weights=[0.8,0.2]`, `k=4` both retrievers**: same scores as
above (confirmed — weighting alone doesn't change the candidate set, only order).

**Hybrid search, `bm25_k=2`, `weights=[0.7,0.3]`**: not yet measured — next step.

## Git / Repo Hygiene (done, as of pre-hybrid-search session)
- Two commits made: initial commit, then removal of accidentally-tracked `.idea/`
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `*.db`,
  `*.db-shm`, `*.db-wal`, `temp_*.pdf`, `evaluation_results.json`, `test.pdf`,
  `.idea/`, `.DS_Store`, `requirements_current.txt`, `.agents/`, `.claude/`
- Git identity set to real name/email
- API key was rotated after being pasted in a chat session — treat any key
  visible outside `.env` as compromised going forward
- **Not yet done as of hybrid-search work**: no new commit made covering the
  hybrid search changes (BM25, EnsembleRetriever, langchain-classic dependency)

## Open Issues / Not Yet Started
- **Hybrid search tuning (current work, in progress)**: re-run `evaluation.py`
  with `bm25_k=2, weights=[0.7,0.3]` and compare against baselines above.
  If it doesn't recover close to the pre-hybrid baseline, consider: lowering
  `k` further, filtering exercise-list-pattern chunks at ingestion time via
  regex, or reverting hybrid search entirely if it's not net-positive for
  this corpus.
- **`requirements.txt` not yet updated** with `langchain-classic` and
  `rank_bm25` — needs confirming both are added
- **Deployment** (Days 19-20) — not started. Streamlit Community Cloud is the
  likely path but not yet decided/executed
- **Resume/portfolio writeup** — not started
- **README.md for the GitHub repo** — doesn't exist yet
- **Other stretch features not started**: LangSmith tracing, LLM-swap
  abstraction, semantic chunking, multi-user auth, caching, async ingestion,
  cost/quota routing — deliberately deferred, hybrid search was picked as
  the first stretch feature to tackle

## Conventions / Rules to Follow
- User is building this from scratch for learning — explain *what* each piece
  of code does and *why*, in small incremental steps, not large code dumps
- User pastes/edits code themselves after explanation; debugging is
  collaborative — read actual tracebacks/output together, don't guess
- Prefer free/local solutions given budget constraints (local embeddings,
  Gemini free tier, DuckDuckGo search)
- Keep `requirements.txt` in sync with every new dependency; hand-curate,
  don't just paste `pip freeze` output
- Streamlit apps run via `streamlit run <file>` in terminal — never PyCharm's
  Run button
- Claude has no direct filesystem access to the user's Mac — all file edits
  are given as exact paste-able content for the user to apply themselves in
  PyCharm; user confirms with "done" after each edit before proceeding
- When investigating a metric/bug, verify claims manually (claim-by-claim,
  reading actual retrieved context) rather than accepting aggregate scores
  or theories at face value — this pattern has repeatedly caught real issues
  (RAGAS faithfulness investigation, and the EnsembleRetriever
  weight-vs-k misunderstanding during hybrid search work)
- Given the 20-day placement-prep timeline, favor pragmatic/simple
  implementations defensible in an interview over premature complexity
- Files mounted in project knowledge/attachments can be stale snapshots from
  earlier sessions — always cross-check against pasted real file contents
  when something looks inconsistent, rather than assuming the mounted
  version is current