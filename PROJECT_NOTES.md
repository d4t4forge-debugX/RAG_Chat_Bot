# RAG Chatbot — Project Notes

Project path: `/Users/hawkeyez007/Desktop/RAG_Chat_Bot`
Goal: Resume-ready RAG chatbot for placement interviews. Python 3.14, PyCharm,
Streamlit frontend, LangGraph backend, Gemini free tier, local embeddings.

## Stack
- LLM: Google Gemini `gemini-3.5-flash-lite` (`thinking_level="low"`), now
  selected via `llm_config.py`'s `get_llm()` factory (see LLM-swap section below)
- Embeddings: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS
- Lexical retrieval: BM25 (`rank_bm25` package, via `langchain_community.retrievers.BM25Retriever`)
- Hybrid retrieval: `EnsembleRetriever` (Reciprocal Rank Fusion) — **now lives in
  `langchain_classic.retrievers`, not `langchain.retrievers`**, as of LangChain v1.0
- Persistence: `SqliteSaver` (LangGraph), db file `chatbot.db`
- Web search: `ddgs` package (renamed from `duckduckgo-search`)
- Evaluation: RAGAS 0.3.9
- Observability: LangSmith tracing (free tier), project name `rag-chatbot`, enabled
  via `.env` vars only (`LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`)
  — no code changes needed for base tracing; `run_name` + `metadata={"thread_id":...}`
  added at both `invoke()` call sites (backend `__main__` smoke test, Streamlit live
  chat) so traces are identifiable per conversation in the dashboard rather than
  generic "LangGraph"
- Caching: simple in-memory dict cache on `rag_tool` retrieval (see Caching section)
- Async ingestion: background-threaded PDF ingestion with live progress (see section below)

## Files
- `langgraph_backend.py` — LangGraph graph: guardrail_node → chat_node ⇄ tools,
  with human_review_node gating web search. Checkpointer, retrieve_all_threads().
  `__main__` smoke test passes `run_name`/`metadata` for LangSmith identifiability.
  LLM now built via `get_llm()` from `llm_config.py` instead of inline construction.
- `streamlit_frontend.py` — Streamlit UI: sidebar threads, PDF upload (now async
  with live progress), HITL approve/reject cards, thread-switch with
  pending-interrupt restoration. Live chat `invoke()` calls pass `run_name`/
  `metadata` for LangSmith identifiability.
- `rag_utils.py` — PDF loading/chunking/cleaning, embeddings, FAISS, BM25, hybrid
  ensemble retriever, per-thread retriever store (`_THREAD_RETRIEVERS`,
  `_THREAD_METADATA` dicts), `ingest_pdf_for_thread()` (sync), `start_ingestion_async()`
  + `get_ingestion_status()` (async, background thread), `get_retriever_for_thread()`,
  `rag_tool` (now with in-memory query caching via `_QUERY_CACHE`).
- `llm_config.py` — **NEW**. Single source of truth for LLM selection: `get_llm()`
  factory function, model/thinking-level overridable via `LLM_MODEL` /
  `LLM_THINKING_LEVEL` env vars, defaults to `gemini-3.5-flash-lite` / `low`.
- `evaluation.py` — standalone RAGAS harness: ingests test.pdf fresh, runs 3
  test questions through the real chatbot, scores with RAGAS, saves
  `evaluation_results.json`. Uses `AnswerRelevancy(strictness=1)` and a `RunConfig`
  (180s timeout, `max_workers=1`) for Gemini-evaluator compatibility/stability.
  Unaffected by the `llm_config.py` refactor — its `from langgraph_backend import llm`
  import still works unchanged.
- `requirements.txt` — curated direct dependencies (not a raw pip freeze).
  Includes `langchain-classic` and `rank_bm25` for hybrid search.
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
- Verified end-to-end in LangSmith trace view: full node tree visible
  (`guardrail_node` → `chat_node` → `tools`/`human_review_node` → `chat_node`),
  including tool call args, per-node token counts, cost, and latency.

## Retrieval architecture (hybrid search — COMPLETE)

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

**Known issue, now resolved**: this specific PDF (Hands-On ML textbook) has many
end-of-chapter "Exercises" sections — short, numbered, keyword-dense questions
covering many ML terms in a small space. BM25 ranked these highly for almost any
ML query even though they're not explanatory content, due to raw keyword density.
Fixed by shrinking `bm25_k` to 2 independent of the ensemble weights (see
debugging trail below).

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
4. Correct fix applied: added independent `bm25_k` parameter (default 2)
   to shrink BM25's candidate pool at the source rather than relying on
   weighting. Manually verified this reduces exercise-list chunks in raw
   retrieval output (from 4/8 to 2/6 for the "What is supervised learning?"
   query).
5. **Re-ran `evaluation.py` with `bm25_k=2, weights=[0.7,0.3]`** — final result:
   - faithfulness: 0.8333 (per-question: [1.0, 1.0, 0.5])
   - answer_relevancy: 0.8116
   - context_precision: 0.5278 (beats pre-hybrid baseline of 0.5111)
6. **Manually verified the low faithfulness score (0.5 on the "regularization"
   question)** by cross-checking the generated answer's three claims against
   retrieved context, sentence by sentence — all three claims were fully
   supported. Concluded this is RAGAS-judge noise (n=3 questions means one
   flagged score swings the average by ~0.17), not a real faithfulness
   regression. **Decision: hybrid search accepted as-is.** context_precision
   and answer_relevancy both recovered to/past baseline; the faithfulness dip
   doesn't survive manual inspection.
7. Later, after the LLM-swap refactor, re-ran `evaluation.py` again as a
   verification step (unrelated to hybrid search) — faithfulness came back as
   0.9333 this time (context_precision stayed identical at 0.5278, confirming
   retrieval logic was untouched). This is further evidence the faithfulness
   metric is simply noisy run-to-run on this small eval set, not tied to any
   particular code change.

## LangSmith Tracing — COMPLETE

Enabled purely via `.env`:

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=<key from smith.langchain.com>
LANGSMITH_PROJECT=rag-chatbot

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

## LLM-Swap Abstraction — COMPLETE

New file `llm_config.py`:
- `get_llm()` factory function is now the single source of truth for which
  model the app uses (main chat, guardrail check, RAGAS evaluator via
  `evaluation.py`'s `from langgraph_backend import llm`)
- Model name / thinking level overridable via `LLM_MODEL` / `LLM_THINKING_LEVEL`
  env vars, defaulting to `gemini-3.5-flash-lite` / `low`
- Rationale: interview-defensible answer to "how would you make this swappable
  if the free tier ran out or you wanted to try a different provider" — change
  one file/env var, not every file that happens to construct an LLM
- Verified: HITL smoke test and full `evaluation.py` run both pass unchanged
  after the swap. RAGAS score movement observed on re-run (faithfulness
  0.83→0.93) attributed to already-documented judge noise on n=3 questions,
  not this change — context_precision (retrieval-only metric) stayed
  identical at 0.5278, confirming retrieval logic was untouched.

## In-Memory Query Caching — COMPLETE

`rag_utils.py`:
- `_QUERY_CACHE` dict, keyed by `(thread_id, normalized query)` (query is
  `.strip().lower()`'d so trivial whitespace/case differences still hit cache)
- `rag_tool` checks cache before hitting the retriever; only successful
  retrievals are cached (errors like "no document uploaded" are never
  cached, so a later successful ingest isn't blocked by a stale cached error)
- Returned dict includes `cache_hit: true/false` for observability/demo
- Verified: ~140x speedup on identical repeated query via standalone script
  (0.0296s uncached vs 0.0002s cached)
- **Known limitation (documented, deliberately not fixed)**: only caches
  the retrieval step, not the final LLM response. End-to-end latency on a
  repeated question is still dominated by the Gemini API call itself
  (1-4s, per LangSmith trace data). Caching LLM output was deliberately
  skipped — it raises staleness/correctness questions (e.g. if the system
  prompt changes) disproportionate to time available. Be ready to explain
  this distinction clearly if asked "how much does this actually save
  end-to-end" — the honest answer is "redundant retrieval work, not
  redundant LLM calls."

## Async PDF Ingestion — COMPLETE

`rag_utils.py` + `streamlit_frontend.py`:
- `start_ingestion_async()` launches ingestion on a real background
  `threading.Thread`; `_ingest_pdf_worker()` updates a shared
  `_INGESTION_STATUS` dict through stages (loading → embedding → building
  retriever → done/error); `get_ingestion_status()` lets the UI poll
  without blocking
- Original synchronous `ingest_pdf_for_thread()` kept unchanged —
  `evaluation.py` and `rag_utils.py`'s own `__main__` block still use it
- Streamlit's "Process PDF" button now starts async ingestion and returns
  immediately; UI polls status every 0.5s and shows live stage messages
  until done/error, then cleans up temp file + session state
- Verified manually: watched live stage transitions in the sidebar
  (Loading PDF → Building embeddings for N chunks → Building hybrid
  retriever → Indexed) rather than one frozen blocking spinner
- **Known scope note (documented, be ready to explain if asked)**:
  Streamlit's execution model reruns the whole script per interaction —
  this isn't true async I/O in the FastAPI/asyncio sense. The real benefit
  here is (a) the background thread doesn't block the main Streamlit
  process, and (b) the user gets live progress instead of a frozen spinner
  — not concurrent multi-user throughput, which doesn't apply to this
  single-session demo app.

**Deferred, deliberately not built**: semantic chunking (would require
redoing the same expensive re-embed → re-RAGAS → manual-verify cycle
hybrid search already went through, and the corpus-noise problem found
during hybrid search tuning is orthogonal to chunking strategy), multi-user
auth, cost/quota routing (lower interview value for a solo portfolio
project, more relevant to actual production multi-tenant systems).

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
- **Stretch features built, in order, and why**:
  1. Hybrid search — demonstrate retrieval depth beyond pure vector search
  2. LangSmith tracing — low implementation risk (pure config), strong
     interview value (observability story)
  3. LLM-swap abstraction — low risk, clean architecture talking point
  4. In-memory caching — moderate effort, strong "production thinking" story
  5. Async ingestion — moderate effort, natural pairing with caching work
  Semantic chunking, multi-user auth, and cost/quota routing were
  deliberately skipped (see reasoning in their respective sections above).
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
    ingestion completes (both sync and now async paths clean up correctly —
    verified no stray `temp_*.pdf` left behind after async ingestion test)
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
19. **Not a bug** — LLM-swap abstraction, caching, and async ingestion all
    worked cleanly on first implementation, verified via smoke test /
    standalone script / manual UI observation respectively. A one-time git
    staging quirk occurred (PyCharm auto-staged a throwaway `test_cache.py`
    verification script before it was deleted) — resolved with
    `git restore --staged` before the real commit; no code issue involved.

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
    Run 1 (right after bm25_k fix):
      faithfulness: 0.8333  [per-question: 1.0, 1.0, 0.5 — low score traced to
                              RAGAS judge noise on n=3, manually verified all
                              claims in flagged answer were faithful to context]
      answer_relevancy: 0.8116
      context_precision: 0.5278  (BEATS pre-hybrid baseline of 0.5111)
    Run 2 (after LLM-swap refactor, sanity-check re-run):
      faithfulness: 0.9333
      answer_relevancy: 0.8756
      context_precision: 0.5278  (identical — confirms retrieval untouched)
    Conclusion: context_precision is stable/reproducible at 0.5278 across
    runs (retrieval logic is deterministic). faithfulness/answer_relevancy
    vary run-to-run due to RAGAS judge noise on a small n=3 eval set — this
    is now confirmed by two independent re-runs, not just one investigation.

## Git / Repo Hygiene
- Seven commits made so far:
  1. Initial commit: RAG chatbot with LangGraph, HITL, RAGAS eval
  2. Remove `.idea/` from tracking (editor config, not project source)
  3. `d306ba3` — Add hybrid search (FAISS + BM25), tune RAGAS eval, tighten
     groundedness prompt (includes `requirements.txt` update, `PROJECT_NOTES.md`
     update, `.gitignore` update to exclude `PROJECT KNOWLEDGE`)
  4. `6dc05de` — Add LangSmith tracing: run names and thread_id metadata
  5. `fe0d7e5` — Update PROJECT_NOTES.md (hybrid search + tracing complete)
  6. `bb7f3ba` — Add LLM-swap abstraction via llm_config.py
  7. `8cc68b8` — Add in-memory query caching to rag_tool
  8. `aaa5e32` — Add async PDF ingestion with live progress feedback
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
- **Other stretch features**: semantic chunking, multi-user auth, cost/quota
  routing — deliberately deferred (see reasoning in their sections above).
  Hybrid search, LangSmith tracing, LLM-swap abstraction, caching, and async
  ingestion are all complete.

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
  and confirmed non-issues (RAGAS faithfulness investigation, the
  EnsembleRetriever weight-vs-k misunderstanding, and two independent
  confirmations of RAGAS judge noise on n=3)
- Before committing, check `git status`/`git diff` for each modified file
  rather than assuming only the intended files changed — this caught an
  untracked `PROJECT KNOWLEDGE` file, confirmed two files with legitimate
  uncommitted changes from an earlier session, and caught a PyCharm
  auto-staged throwaway test script before it could be committed
- Given the 20-day placement-prep timeline, favor pragmatic/simple
  implementations defensible in an interview over premature complexity.
  Don't chase evaluator noise (e.g. single-question RAGAS score swings)
  when manual verification already confirms correctness. When choosing
  which stretch features to build, weigh effort vs. interview value vs.
  risk of turning into a multi-day debugging saga (this is why semantic
  chunking was deferred and LangSmith/LLM-swap were prioritized first).
- Files mounted in project knowledge/attachments can be stale snapshots from
  earlier sessions — always cross-check against pasted real file contents
  when something looks inconsistent, rather than assuming the mounted
  version is current
- When the user says to proceed through a batch of items "one by one" or
  "all of the above," do not stop to ask which one to do next between each
  item — just proceed through the full sequence, one step at a time within
  each item, only pausing for the user's "done" confirmation on individual
  edits, not for direction-choosing between items.

## Continue from here
Five stretch features are complete, evaluated/verified, and committed:
hybrid search (`d306ba3`), LangSmith tracing (`6dc05de`), LLM-swap
abstraction (`bb7f3ba`), query caching (`8cc68b8`), async ingestion
(`aaa5e32`). Next session should move to **deployment** (Days 19-20) —
likely Streamlit Community Cloud, not yet decided/executed — followed by
README.md and resume writeup. No open technical debt or unresolved bugs
as of this session's end.

That's the complete file. Once you've pasted it in and saved, let me know and we'll do the final commit for this session, then it's safe to compact — a fresh chat can pick up entirely from this file without you re-explaining anything.