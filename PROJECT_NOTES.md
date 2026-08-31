`langsmith` package already present as a transitive dependency of `langchain-core`
(not added explicitly to `requirements.txt` since nothing imports it directly).

No code changes needed for tracing itself — LangGraph auto-instruments node names
straight from the graph definition. Added `run_name` and `metadata={"thread_id":...}`
at the two `chatbot.invoke()` call sites (backend smoke test, Streamlit live chat)
purely for dashboard identifiability — cosmetic, not functional.

Verified in dashboard: trace tree shows real node names (`guardrail_node`,
`chat_node`, `route_after_guardrail`, `route_after_chat`, `human_review_node`,
`tools`), tool call args (e.g. `duckduckgo_search` query text, `calculator` args),
per-node token/cost/latency, and the full HITL interrupt → rejection → retry
sequence end-to-end.

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
  tuning — this is the accepted final value; the k=6 pre-hybrid decision is
  effectively superseded by hybrid search's own tuning (k=4 FAISS + bm25_k=2).
- **Hybrid search added as first stretch feature** (Days 20 roadmap explicitly
  deferred this) to demonstrate retrieval depth beyond pure vector search.
  Completed, evaluated, and committed.
- **LangSmith tracing added as second stretch feature** — chosen over semantic
  chunking / LLM-swap / caching because of low implementation risk (pure config,
  no re-architecture) and strong interview value (concrete answer to "how do you
  observe/debug your pipeline in production").
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
18. **Not a bug** — LangSmith tracing integration worked cleanly on the first
    attempt. Noted here to confirm nothing broke: verified via dashboard that
    the full node tree, tool call args, and HITL interrupt/rejection flow all
    traced correctly with zero code changes needed for base tracing.

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

**Hybrid search, `weights=[0.5,0.5]`, `k=4` both retrievers** (regressed —
root-caused to BM25 noise, see "Retrieval architecture" section above):
    faithfulness: 0.8561
    answer_relevancy: 0.7598
    context_precision: 0.3921

**Hybrid search, `weights=[0.8,0.2]`, `k=4` both retrievers**: same scores as
above (confirmed — weighting alone doesn't change the candidate set, only order).

**Hybrid search, `bm25_k=2`, `weights=[0.7,0.3]` (FINAL, accepted config)**:
    faithfulness: 0.8333  [per-question: 1.0, 1.0, 0.5 — low score traced to
                            RAGAS judge noise on n=3, not a real error; manually
                            verified all claims in the flagged answer were
                            faithful to retrieved context]
    answer_relevancy: 0.8116  (near-full recovery to pre-hybrid baseline)
    context_precision: 0.5278  (BEATS pre-hybrid baseline of 0.5111)

## Git / Repo Hygiene
- Four commits made so far:
  1. Initial commit: RAG chatbot with LangGraph, HITL, RAGAS eval
  2. Remove `.idea/` from tracking (editor config, not project source)
  3. `d306ba3` — Add hybrid search (FAISS + BM25), tune RAGAS eval, tighten
     groundedness prompt (includes `requirements.txt` update, `PROJECT_NOTES.md`
     update, `.gitignore` update to exclude `PROJECT KNOWLEDGE`)
  4. `6dc05de` — Add LangSmith tracing: run names and thread_id metadata
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `*.db`,
  `*.db-shm`, `*.db-wal`, `temp_*.pdf`, `evaluation_results.json`, `test.pdf`,
  `.idea/`, `.DS_Store`, `requirements_current.txt`, `.agents/`, `.claude/`,
  `PROJECT KNOWLEDGE` (chat-session-attachment doc, not source — regenerated
  fresh each session, not meant to be versioned)
- Git identity set to real name/email
- API key was rotated after being pasted in a chat session — treat any key
  visible outside `.env` as compromised going forward
- `requirements.txt` confirmed complete: includes `langchain-classic` and
  `rank_bm25` (both verified installed via `pip show` before adding)

## Open Issues / Not Yet Started
- **Deployment** (Days 19-20) — not started. Streamlit Community Cloud is the
  likely path but not yet decided/executed. Note: will need the `ragas` manual
  patch step documented above if evaluation is ever run in the deployed
  environment (unlikely — eval is a local dev tool, not part of the deployed app).
- **Resume/portfolio writeup** — not started
- **README.md for the GitHub repo** — doesn't exist yet
- **Other stretch features not started**: LLM-swap abstraction, semantic
  chunking, multi-user auth, caching, async ingestion, cost/quota routing —
  deliberately deferred. Hybrid search and LangSmith tracing are both complete.

## Conventions / Rules to Follow
- User is new to practical implementation and wants every step spelled out
  explicitly: exact terminal commands, exact PyCharm click-paths (open file →
  Cmd+F to find → what to select → what to paste → Cmd+S to save), one step
  at a time, waiting for "done" before proceeding to the next step. Don't
  assume familiarity with IDE shortcuts or terminal basics.
- Explain *what* each piece of code does and *why*, in small incremental
  steps, not large code dumps
- User pastes/edits code themselves after explanation; debugging is
  collaborative — read actual tracebacks/output together, don't guess
- Prefer free/local solutions given budget constraints (local embeddings,
  Gemini free tier, DuckDuckGo search, LangSmith free tier)
- Keep `requirements.txt` in sync with every new dependency; hand-curate,
  don't just paste `pip freeze` output. Verify packages are actually
  installed via `pip show` before adding lines, rather than assuming.
- Streamlit apps run via `streamlit run <file>` in terminal — never PyCharm's
  Run button
- Claude has no direct filesystem access to the user's Mac — all file edits
  are given as exact paste-able content for the user to apply themselves in
  PyCharm; user confirms with "done" after each edit before proceeding
- When investigating a metric/bug, verify claims manually (claim-by-claim,
  reading actual retrieved context) rather than accepting aggregate scores
  or theories at face value — this pattern has repeatedly caught real issues
  (RAGAS faithfulness investigation, the EnsembleRetriever weight-vs-k
  misunderstanding, and the n=3 faithfulness-noise investigation)
- Before committing, check `git status`/`git diff` for each modified file
  rather than assuming only the intended files changed — this caught an
  untracked `PROJECT KNOWLEDGE` file and confirmed two files
  (`evaluation.py`, `langgraph_backend.py`) had legitimate uncommitted
  changes from an earlier session that matched documented bug fixes
- Given the 20-day placement-prep timeline, favor pragmatic/simple
  implementations defensible in an interview over premature complexity.
  Don't chase evaluator noise (e.g. single-question RAGAS score swings)
  when manual verification already confirms correctness.
- Files mounted in project knowledge/attachments can be stale snapshots from
  earlier sessions — always cross-check against pasted real file contents
  when something looks inconsistent, rather than assuming the mounted
  version is current

## Continue from here
Hybrid search and LangSmith tracing are both complete, evaluated/verified,
and committed (commits `d306ba3` and `6dc05de`). Next session should pick
one of: (1) another stretch feature from the deferred list above, (2)
deployment (Days 19-20), or (3) README.md / resume writeup. No open
technical debt or unresolved bugs as of this session's end.