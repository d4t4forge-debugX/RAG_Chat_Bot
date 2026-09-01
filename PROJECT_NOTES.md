Here's the complete, corrected `PROJECT_NOTES.md` — merged from both versions you pasted (doc 21 was the more current/accurate one; I used it as the base and folded in anything doc 22 had that wasn't stale), plus everything from today's session added in the right places. This is the full file, not a diff — replace your entire `PROJECT_NOTES.md` with this.

```markdown
# RAG Chatbot — Project Notes

Project path: `/Users/hawkeyez007/Desktop/RAG_Chat_Bot`
Goal: Resume-ready RAG chatbot for placement interviews. Python 3.14, PyCharm,
Streamlit frontend, LangGraph backend, Gemini free tier, local embeddings.
Deployed on Streamlit Cloud.
GitHub: github.com/d4t4forge-debugX/RAG_Chat_Bot

## Stack (confirmed against current requirements.txt / llm_config.py)
- LLM: Google Gemini via `llm_config.py`'s `get_llm()` factory — defaults to
  `gemini-3.5-flash-lite` / `thinking_level="low"`, overridable via
  `LLM_MODEL` / `LLM_THINKING_LEVEL` env vars (`ChatGoogleGenerativeAI`)
- Embeddings: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS (`faiss-cpu`)
- Lexical retrieval: BM25 (`rank_bm25`, via `langchain_community.retrievers.BM25Retriever`)
- Hybrid retrieval: `EnsembleRetriever` from `langchain_classic.retrievers`
  (moved out of core `langchain` in LangChain v1.0 — `langchain-classic` is a
  direct dependency in requirements.txt because of this)
- Persistence: `SqliteSaver` (`langgraph-checkpoint-sqlite`), db file `chatbot.db`
- Web search: `ddgs` package (renamed from `duckduckgo-search`)
- Evaluation: `ragas==0.3.9` (pinned — see manual patch note below), plus
  `langchain-google-vertexai` (needed for the patch's fallback import)
- Observability: LangSmith tracing via `.env` vars only
- Caching: in-memory dict cache on `rag_tool`, keyed by `(thread_id, query)`
- Async ingestion: background `threading.Thread`-based PDF ingestion with
  live progress polling from the Streamlit UI

## Files (roles confirmed against current code in context)
- **`llm_config.py`** — single source of truth for LLM selection. `get_llm()`
  reads `LLM_MODEL`/`LLM_THINKING_LEVEL` from env, falls back to
  `gemini-3.5-flash-lite`/`low`. Used by `langgraph_backend.py` (main chat +
  guardrail check) and `evaluation.py` (via `from langgraph_backend import llm`).
- **`langgraph_backend.py`** — graph: `guardrail_node` → `chat_node` ⇄ `tools`,
  with `human_review_node` gating only `duckduckgo_search`. Builds `llm` via
  `get_llm()`. `ChatState` = `{messages: Annotated[list[BaseMessage], add_messages], blocked: bool}`.
  Tools bound: `search_tool` (DuckDuckGo), `calculator`, `rag_tool` (imported
  from `rag_utils`). `checkpointer` = `SqliteSaver` on `chatbot.db`.
  `retrieve_all_threads()` helper. `__main__` block runs a HITL smoke test:
  invokes with a query that should trigger web search, then loops over
  `result["__interrupt__"]` prompting approve/reject via terminal input until
  no interrupt remains.
  **Cosmetic note (not yet fixed)**: `tools_condition` is imported from
  `langgraph.prebuilt` but never used — replaced by custom `route_after_chat`/
  `route_after_guardrail`. Harmless, listed under cleanup tasks below.
- **`streamlit_frontend.py`** — sidebar: New Chat button, PDF uploader +
  "Process PDF" button wired to `start_ingestion_async`, live polling of
  `get_ingestion_status` (shows stage text, `st.rerun()`s every 0.5s while
  running), thread list with switch-and-restore (`load_conversation` returns
  `(messages, pending_interrupt)` so a paused approval survives a thread
  switch). Main area: renders history, then either an Approve/Reject card
  (`st.session_state["pending_interrupt"]`) or a `st.chat_input`. Currently
  uses plain `chatbot.invoke()` (not `.stream()`) both for normal turns and
  for resuming via `Command(resume="approve"/"reject")` — **streaming re-add
  is in progress, see "Open Issues" below.**
  `cleanup_finished_ingestions()` (renamed from the old `clear_stale_ingestion()`
  — see Bug 20 below) sweeps temp files for any thread whose ingestion has
  actually finished, regardless of which thread is currently active.
- **`rag_utils.py`** — `load_and_split_pdf()` (PyPDFLoader +
  `RecursiveCharacterTextSplitter`, chunk_size=1000/overlap=200, plus
  `clean_text()` to strip orphaned unicode surrogates), `build_vector_store()`
  (FAISS), `build_bm25_retriever()`, `get_retriever()` (builds the hybrid
  `EnsembleRetriever`, FAISS `k=4` + BM25 `bm25_k=2`, `weights=[0.7, 0.3]`).
  Per-thread state: `_THREAD_RETRIEVERS`, `_THREAD_METADATA` dicts (in-memory,
  reset on restart) — **note: `_THREAD_RETRIEVERS[thread_id] = ...` overwrites
  on re-ingestion, so only one PDF is tracked per thread at a time; not yet
  confirmed whether this is an intentional design choice or worth changing,
  see "Open Issues" below.** `ingest_pdf_for_thread()` — sync version, used by
  `evaluation.py` and this file's own `__main__` block. `_ingest_pdf_worker()`
  + `start_ingestion_async()` + `get_ingestion_status()` — background-thread
  ingestion with a shared `_INGESTION_STATUS` dict (keyed by thread_id) for
  polling. `get_retriever_for_thread()`. `_QUERY_CACHE` dict +
  `_clear_thread_cache()` (invalidates cache for a thread on re-ingestion).
  `rag_tool` — the LLM-callable tool: checks cache first, else retrieves,
  caches only successful results, returns `cache_hit: true/false` in output.
- **`evaluation.py`** — standalone RAGAS harness. Ingests `test.pdf` fresh
  into a throwaway `thread_id`, runs 3 fixed test questions through the real
  `chatbot`, scores with `Faithfulness()`, `AnswerRelevancy(strictness=1)`,
  `LLMContextPrecisionWithReference()` using a `RunConfig(timeout=180,
  max_workers=1)` for Gemini-evaluator stability. Saves `evaluation_results.json`.
- **`requirements.txt`** — hand-curated. Notable entries beyond the obvious:
  `langchain-classic` (for `EnsembleRetriever`), `rank_bm25`, `ragas==0.3.9`
  (pinned — required for the manual patch below), `langchain-google-vertexai`
  (fallback import target for that same patch).
- `test.pdf` — Hands-On Machine Learning by Aurélien Géron (gitignored, not
  committed — copyrighted).

## Architecture (current graph shape, confirmed in langgraph_backend.py)

```
START → guardrail_node → (conditional: route_after_guardrail)
  ├─ blocked=True  → END (refusal AIMessage)
  └─ blocked=False → chat_node → (conditional: route_after_chat)
       ├─ tool_calls[0].name == "duckduckgo_search" → human_review_node
       │     → interrupt() → decision == "approve" → Command(goto="tools") → chat_node (loop)
       │     → decision != "approve" → Command(goto="chat_node", update=denial ToolMessage)
       ├─ any other tool_call → tools → chat_node
       └─ no tool_calls → END
```

- `route_after_guardrail` checks `state.get("blocked")`.
- `guardrail_node` only runs the appropriateness check when the last message
  is a `HumanMessage`; uses `is_query_appropriate()`, a single permissive
  LLM call (`llm.invoke(check_prompt)`, expects YES/NO).
- `chat_node` builds a `SystemMessage` per-invocation that injects the
  current `thread_id` (from `config["configurable"]["thread_id"]`) so the
  model knows what to pass to `rag_tool`. System prompt also instructs:
  multi-step sub-question breakdown for complex queries, explicit
  "doesn't appear to cover this topic" fallback when retrieval is
  insufficient, and a hard rule against naming methods/concepts not present
  in retrieved context (groundedness guardrail).
- Only `search_tool` is gated behind human approval — sole tool making an
  external/untrusted network call. `calculator` and `rag_tool` are
  deterministic/internal, no approval needed.

## Key Decisions & Why
- **Local embeddings over OpenAI**: no per-call cost, strong interview talking point.
- **Gemini free tier + `thinking_level="low"`**: budget constraint drove both choices.
- **Per-thread RAG state kept in-memory** (`_THREAD_RETRIEVERS`/`_THREAD_METADATA`):
  simple, per-conversation PDF isolation; known limitation — resets on app
  restart/redeploy, accepted and documented rather than solved (would need a
  persistent store for real multi-session durability).
- **HITL gates only web search, not calculator/rag_tool**: same rationale
  repeated in code comments — only tool touching an untrusted external source.
- **Guardrails are prompt-based, not separate classifiers**: input guardrail
  is one cheap LLM call short-circuiting to END; output groundedness is
  system-prompt instruction, not a post-hoc verification step. Simpler to
  build and explain than a dedicated groundedness model.
- **Hybrid retrieval tuning (FAISS k=4, BM25 bm25_k=2, weights=[0.7,0.3])**:
  `EnsembleRetriever` merges the *union* of both retrievers' results and only
  re-ranks by weight — it does not drop candidates based on weight. Confirmed
  by testing: identical result sets appeared under `weights=[0.5,0.5]` and
  `weights=[0.8,0.2]` with equal per-retriever `k`; only order changed. The
  actual fix for BM25 noise (this PDF's dense "Exercises" sections dominating
  keyword search) was shrinking `bm25_k` independently, not reweighting.
  **Re-verified locally (see Verification Log below) — still produces 2/6
  exercise-list chunks for "What is supervised learning?", matching the
  originally documented ratio exactly.**
- **Caching only the retrieval step, not final LLM output** (`rag_tool`'s
  `_QUERY_CACHE`): caching LLM responses raises staleness/correctness
  questions (e.g. system prompt changes) disproportionate to benefit;
  deliberately out of scope.
- **`requirements.txt` hand-curated, not `pip freeze`**: direct dependencies
  only, mostly unpinned except `ragas==0.3.9` (must stay pinned for the patch
  below to keep applying cleanly).
- **Async ingestion status is tracked per-thread_id, not per-browser-session**
  (fixed today — see Bug 20 below): the background worker already keyed
  `_INGESTION_STATUS` by thread_id correctly; the bug was that the Streamlit
  UI was only *reading* that status while a single session-wide flag was
  set, so navigating away lost visibility even though the job kept running
  correctly in the background.

## Bugs Fixed (cumulative)
1. `message`/`messages` key typo in early `ChatState`.
2. Tools referenced before definition (Python top-to-bottom execution order).
3. `duckduckgo-search` → `ddgs` package rename.
4. Broken Unicode surrogates breaking the embedding tokenizer — fixed via
   `re.sub(r'[\ud800-\udfff]', '', text)` in `rag_utils.clean_text()`.
5. Gemini model deprecations — settled on `gemini-3.5-flash-lite`.
6. Free-tier quota exhaustion (429) on a heavier model — switched down.
7. `ragas` import crash — see manual patch section below.
8. HITL: LLM retried rejected tool calls as if the search had merely failed —
   fixed by making the denial `ToolMessage` explicit ("do not retry this tool
   or attempt a similar search"), and by looping over interrupts in both the
   terminal `__main__` test and the Streamlit handler instead of assuming
   exactly one interrupt per turn.
9. Streamlit thread-switch staleness — sidebar thread buttons updated
   `session_state` but didn't call `st.rerun()`; fixed.
10. Debug print flooding terminal — removed a stray print from the result handler.
11. Leftover temp PDF files never deleted — added `os.remove()` cleanup for
    both sync and async ingestion paths, plus (at the time) `clear_stale_ingestion()`
    for the case where the user navigates away mid-ingestion (superseded by
    Bug 20's fix, see below).
12. Duplicate dead code in `__main__`'s final-answer print logic — removed.
13. Leaked API key caught before real damage — key rotated, `.env` untracked
    via `git rm --cached -f`.
14. `.idea/` and a 58MB `test.pdf` committed despite `.gitignore` — removed
    via `git rm --cached` after the fact.
15. RAGAS faithfulness anomaly (0.27) — root cause was the model naming
    methods/concepts not present in retrieved context (e.g. naming
    Ridge/Elastic Net when only Lasso was retrieved). Fixed via k=4→6
    retrieval (later superseded by hybrid tuning) + a tightened system
    prompt forbidding unretrieved terms. Separately fixed a RAGAS/Gemini
    incompatibility (no multi-candidate generation support) via
    `AnswerRelevancy(strictness=1)`.
16. `ingest_pdf_for_thread` broke after `get_retriever`'s signature changed to
    require a `chunks` param (for BM25) — updated the call site.
17. `ModuleNotFoundError: langchain.retrievers` — `EnsembleRetriever` moved to
    the new `langchain-classic` package in LangChain v1.0. Fixed via
    `pip install langchain-classic` + import path change. `BM25Retriever`
    stayed in `langchain_community.retrievers`, unaffected.
18. Async PDF ingestion status ambiguity on the deployed app — investigated
    (see Bug 20, this was the actual root cause, found and fixed this session).
19. LangSmith tracing, LLM-swap abstraction, caching, and async ingestion all
    worked cleanly on first implementation (not bugs, noted for completeness).
    A one-time git staging quirk occurred (PyCharm auto-staged a throwaway
    `test_cache.py` verification script before it was deleted) — resolved
    with `git restore --staged` before the real commit; no code issue involved.
20. **Async ingestion progress lost on thread switch — found and fixed this
    session.** Root cause: `streamlit_frontend.py` tracked "should I poll for
    ingestion progress" using a single session-wide flag,
    `st.session_state["ingesting_temp_path"]` (one string, not scoped per
    thread). Clicking "New Chat" or switching to a different sidebar thread
    called `clear_stale_ingestion()`, which deleted that flag unconditionally
    — even if the ingestion job for the *original* thread was still running
    correctly in the background. Once the flag was gone, the UI never polled
    `get_ingestion_status()` for that thread again, so the user saw no
    progress at all when navigating back — just a sudden "Document loaded"
    once the job happened to finish, with no visible progress in between.
    **Confirmed via manual testing** that the background thread itself was
    *not* actually broken — `PyPDFLoader` reads the whole PDF into memory
    near-immediately, before the slow embedding step, so deleting the temp
    file a few seconds later (when the user clicks away) doesn't crash the
    worker; ingestion completes correctly regardless. The bug was pure lost
    UI visibility, not data loss or a failed job.
    **Fix applied**: replaced the single flag with a per-thread dict,
    `st.session_state["ingesting_temp_paths"]` (`{thread_id: temp_path}`).
    Renamed `clear_stale_ingestion()` → `cleanup_finished_ingestions()`,
    which now loops over all tracked threads and only deletes a temp file /
    stops tracking a thread once `get_ingestion_status(thread_id)` confirms
    that thread's job is actually `"done"` or `"error"` — never based on
    navigation alone. The sidebar's status display now calls
    `get_ingestion_status(current_thread_id)` directly on every rerun,
    independent of any session flag, so it correctly shows live "⏳ stage..."
    messages even after navigating away and back mid-ingestion.
    Also fixed in the same pass: temp filenames now include `thread_id`
    (`temp_{thread_id}_{filename}`) to prevent two threads uploading a
    same-named PDF from colliding on the same temp file path.
    **Verified working**: manually tested the exact repro (upload PDF →
    click Process PDF → click New Chat mid-ingestion → click back into the
    original thread) — live "⏳" progress messages now correctly continue
    displaying after navigating back, instead of showing nothing until a
    sudden "done."

## Manual patch required after any fresh venv rebuild
`ragas==0.3.9` unconditionally imports `langchain_community.chat_models.vertexai`,
a module path that no longer exists in current `langchain-community`. Breaks
any script importing `ragas` (i.e. `evaluation.py`) with:
```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```
**This is a manual edit to an installed package — `requirements.txt` cannot
capture it. Must be reapplied after every fresh `pip install`.**

File to patch: `.venv/lib/python3.14/site-packages/ragas/llms/base.py`
Fix: wrap the `ChatVertexAI`/`VertexAI` imports in try/except, falling back to
`langchain_google_vertexai` (already a pinned dependency), and filter `None`
out of `MULTIPLE_COMPLETION_SUPPORTED` so downstream `isinstance()` checks
don't break.

After patching, clear `ragas`'s `__pycache__` (a stale `.pyc` can mask the fix):
```
find .venv -path "*/ragas/**/__pycache__" -exec rm -rf {} +
```

## RAGAS Baselines (for comparison across experiments)

**Pre-hybrid-search baseline** (k=6, FAISS-only, tightened prompt, strictness=1):
```
faithfulness: 0.9583
answer_relevancy: 0.8203
context_precision: 0.5111
```

**Hybrid search, `weights=[0.5,0.5]`, `k=4` both retrievers** (regressed —
root-caused to BM25 noise):
```
faithfulness: 0.8561
answer_relevancy: 0.7598
context_precision: 0.3921
```

**Hybrid search, `weights=[0.8,0.2]`, `k=4` both retrievers**: same scores as
above (confirmed — weighting alone doesn't change the candidate set, only order).

**Hybrid search, `bm25_k=2`, `weights=[0.7,0.3]` (FINAL, accepted config)**:
```
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
```
Conclusion: context_precision is stable/reproducible at 0.5278 across
runs (retrieval logic is deterministic). faithfulness/answer_relevancy
vary run-to-run due to RAGAS judge noise on a small n=3 eval set — confirmed
by two independent re-runs, not just one investigation.

## Verification Log — local re-verification pass (this session)

All 5 documented stretch features were re-tested locally, one at a time, to
confirm nothing regressed and to investigate the previously-unresolved
async-ingestion question. Results:

1. **Async PDF ingestion** — found and fixed a real bug (Bug 20 above).
   Re-tested after the fix: confirmed live progress now survives switching
   threads or clicking "New Chat" mid-ingestion. ✅
2. **Hybrid search** — ran `python rag_utils.py` (its own `__main__` block).
   2,339 chunks from `test.pdf`. Query "What is supervised learning?"
   returned 6 results: 4 genuine explanatory chunks, 2 exercise-list chunks
   — exactly the documented 2/6 ratio from the original `bm25_k=2` tuning
   work. No regression. ✅
3. **LangSmith tracing** — sent a live chat message locally, confirmed a new
   trace appeared in the `rag-chatbot` LangSmith dashboard project under the
   run name `streamlit_chat_turn`. Also incidentally confirmed multiple other
   trace types are healthy: `hitl_smoke_test`, `ragas evaluation`,
   `EnsembleRetriever` (nested retriever spans), `LangGraph` (parent runs).
   One observation, not a bug: a plain "hello" message showed 17.67s latency
   in one trace vs. ~4-5.5s for real RAG turns — likely a Gemini free-tier
   cold-start/rate-limit blip, not investigated further, worth being aware
   of if it comes up in a live demo. ✅
4. **LLM-swap abstraction** — tested via
   `LLM_MODEL=gemini-2.5-flash python langgraph_backend.py`. This produced a
   `404 NOT_FOUND` error from Google's API, but the error message explicitly
   named `gemini-2.5-flash` as the model it tried to call — which **proves**
   the env-var override correctly reached the real API call. The failure
   itself is external: Google has fully retired `gemini-2.5-flash`
   ("no longer available to new users... use models/gemini-3.6-flash").
   Re-ran with no override afterward (`python langgraph_backend.py` plain) —
   confirmed the default (`gemini-3.5-flash-lite`) still works, full HITL
   reject flow completed correctly. Both the override path and the fallback
   path are verified working. ✅
5. **In-memory query caching** — standalone script (`test_cache.py`,
   deleted after use) called `rag_tool` twice with the same
   `(thread_id, query)`. First call: `cache_hit: False`, 0.0348s. Second
   call: `cache_hit: True`, 0.0002s. **166.6x speedup** — consistent with
   the originally documented ~140x figure (minor run-to-run timing variance
   expected, not a discrepancy). ✅

**Conclusion: all 5 stretch features confirmed working correctly, matching
documented behavior. One real bug found and fixed (async ingestion progress
tracking). No other regressions found.**

## Streaming investigation (this session — diagnostic only, not yet implemented)

Per an earlier session's note, token streaming (`chatbot.stream()`) was
deliberately dropped when HITL/`interrupt()` was first added, to avoid
tackling two hard problems at once. It was never re-added. This was
identified this session as the one genuine, previously-flagged gap between
the current app and a more polished version of it (confirmed not present
anywhere else by cross-checking against course-reference files — see below).

**Diagnostic work done (via a throwaway `test_streaming.py`, since deleted)**:
1. Ran `chatbot.stream(..., stream_mode="messages")` on a query that
   triggers a `duckduckgo_search` tool call (and therefore an interrupt).
   Confirmed:
   - The **guardrail's internal YES/NO check leaks into the raw stream** —
     `is_query_appropriate()`'s `llm.invoke(check_prompt)` call produces its
     own streamed chunks (e.g. `content=[{'type': 'text', 'text': 'YES', ...}]`)
     that must be filtered out before reaching the user.
   - **`metadata['langgraph_node']` reliably distinguishes** `"guardrail_node"`
     vs `"chat_node"` chunks — confirmed via a second test run printing
     `metadata.get('langgraph_node')` per chunk. This gives a clean filter:
     only stream chunks where `metadata['langgraph_node'] == "chat_node"`.
   - **Gemini streams `content` as a list of dicts**, not a plain string —
     e.g. `[{'type': 'text', 'text': '...', 'index': 0}]`, sometimes with an
     `'extras': {'signature': '...'}` block (internal "thinking" trace
     metadata, must be ignored/not displayed). The existing `extract_text()`
     helper in `streamlit_frontend.py` already handles this shape for
     non-streaming responses; the same logic needs a per-chunk version.
   - **The stream does not announce an interrupt** — it simply stops
     yielding once the graph pauses at `human_review_node`. The only
     reliable way to detect a pending interrupt after streaming is to call
     `chatbot.get_state(config)` afterward and check `state.tasks` for
     `task.interrupts` — the same pattern `load_conversation()` already
     uses elsewhere in the app.

**Implementation plan (agreed, not yet applied to `streamlit_frontend.py`)**:
1. Add an `extract_stream_chunk_text(content)` helper (list-of-dicts →
   joined text, mirrors `extract_text()`).
2. Replace the plain-chat-turn branch's `chatbot.invoke()` call with a
   generator function fed into `st.write_stream()`, filtering chunks to only
   `metadata['langgraph_node'] == "chat_node"` and yielding extracted text.
3. After the stream loop ends, call `chatbot.get_state(config)` and check
   for a pending interrupt exactly as above; set
   `st.session_state["pending_interrupt"]` if found, else append the
   accumulated streamed text to `message_history` as normal.
4. **Scope decision**: leave the Approve/Reject resume path
   (`chatbot.invoke(Command(resume=...), config=CONFIG)`) as plain
   `.invoke()` for now, not streamed — the resumed response is typically
   short, and mixing a streamed first-turn with a non-streamed resume is a
   reasonable, explainable scope boundary. Can revisit later for full
   consistency.
5. Tool-status badge (the "🔧 Using `tool_name`..." indicator, sourced from
   a reference tutorial pattern) — planned as a follow-up addition after
   plain text streaming is confirmed working in the browser, not bundled
   into the same pass, to keep changes small and independently testable.

**Not yet done**: none of the above has been applied to the real
`streamlit_frontend.py` yet — this is purely the diagnostic + design phase,
ready to implement next session.

## Reference material clarification (not part of this project)
Files like `langgraph_mcp_backend.py`, `streamlit_frontend_mcp.py`,
`langgraph_tool_backend.py`, `streamlit_frontend_tool.py`,
`langgraph_database_backend.py`, `streamlit_frontend_database.py`,
`streamlit_frontend_threading.py`, `streamlit_frontend_streaming.py`,
`langraph_rag_backend.py`, `streamlit_rag_frontend.py` are **not part of this
project** — they're course/tutorial reference files from a YouTube
tutor (Nitish), used only as a comparison point to check for missing
features. Confirmed via a hardcoded path in one of them
(`/Users/nitish/Desktop/mcp-math-server/main.py` — not Rohit's username/path).
Cross-checking against them surfaced the streaming/tool-status-badge gap
above; nothing else in them applies to this project (MCP integration, raw
tool-calling patterns, and DB-backend variants are already superseded by
this project's more advanced HITL + RAG + hybrid-search implementation).

## Git / Repo Hygiene
- Confirmed current GitHub repo contents (checked via screenshot this
  session) match the real project file list exactly: `.gitignore`,
  `PROJECT_NOTES.md`, `evaluation.py`, `langgraph_backend.py`,
  `llm_config.py`, `rag_utils.py`, `requirements.txt`, `streamlit_frontend.py`,
  plus a `.devcontainer` folder (added separately, "Added Dev Container
  Folder" commit). No stray/leftover files, no missing files. 10 commits so
  far as of last check.
- Nine commits (roughly, per earlier session's log — recount before next
  push):
  1. Initial commit: RAG chatbot with LangGraph, HITL, RAGAS eval
  2. Remove `.idea/` from tracking
  3. `d306ba3` — Add hybrid search (FAISS + BM25), tune RAGAS eval, tighten
     groundedness prompt
  4. `6dc05de` — Add LangSmith tracing: run names and thread_id metadata
  5. `fe0d7e5` — Update PROJECT_NOTES.md (hybrid search + tracing complete)
  6. `bb7f3ba` — Add LLM-swap abstraction via llm_config.py
  7. `8cc68b8` — Add in-memory query caching to rag_tool
  8. `aaa5e32` — Add async PDF ingestion with live progress feedback
  9. `9212dd5` — Added Dev Container Folder
  10. **Not yet committed**: today's async-ingestion thread-switch fix
      (Bug 20) — still only local, needs to be pushed. This is the next real
      commit to make.
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `*.db`,
  `*.db-shm`, `*.db-wal`, `temp_*.pdf`, `evaluation_results.json`, `test.pdf`,
  `.idea/`, `.DS_Store`, `requirements_current.txt`, `.agents/`, `.claude/`,
  `PROJECT KNOWLEDGE` (chat-session-attachment doc, not source)
- Git identity set to real name/email
- API key was rotated after being pasted in a chat session — treat any key
  visible outside `.env` as compromised going forward
- `requirements.txt` confirmed complete: includes `langchain-classic` and
  `rank_bm25` (both verified installed via `pip show` before adding)
- **Reminder pattern**: this session used two throwaway verification
  scripts, `test_cache.py` and `test_streaming.py`, both deleted after use.
  Confirm via `git status` before the next commit that neither got
  auto-staged by PyCharm (same quirk as a previous session).

## Open Issues / Not Yet Started

**Final priority list, in order (deployment deliberately last per Rohit's
instruction)**:

1. **Streaming + live tool-status badge — IN PROGRESS.** Diagnostic phase
   complete (see "Streaming investigation" above), full implementation plan
   agreed, not yet applied to `streamlit_frontend.py`. Next session should
   start here: add the `extract_stream_chunk_text()` helper first, then the
   streaming generator, then the post-stream interrupt check, test in
   browser, then add the tool-status badge as a separate follow-up step.
2. **Automated tests (pytest)** — not started. Planned scope: `rag_tool`
   caching logic (cache hit/miss, cache invalidation on re-ingestion),
   `is_query_appropriate()`'s YES/NO parsing, possibly the `calculator` tool.
3. **Single-PDF-per-thread behavior — investigate and decide.**
   `_THREAD_RETRIEVERS[thread_id] = retriever` overwrites on each
   ingestion — uploading a second PDF to the same thread silently replaces
   the first one's retriever rather than adding to it. Not yet determined
   whether this should be treated as an intentional design decision (simpler,
   "one document per conversation," easy to document and defend) or
   changed to support multiple documents per thread. Decide and either fix
   or explicitly document the reasoning.
4. **README.md for the GitHub repo** — doesn't exist yet.
5. **Resume/portfolio writeup** — not started.
6. **Cosmetic cleanup** — remove the unused `tools_condition` import in
   `langgraph_backend.py` (harmless, noted during file review, trivial fix,
   can be folded into any future commit).
7. **Deployment (Days 19–20) — deliberately last.** Push all pending local
   changes (the Bug 20 async-ingestion fix is the main one) to GitHub,
   redeploy to Streamlit Cloud, then re-verify all 5 stretch features on the
   *live* deployed app the same way they were just verified locally this
   session (the local verification pass was a stand-in/prerequisite for
   this, not a replacement).

**Deliberately deferred, not gaps** (unchanged from earlier sessions):
semantic chunking, multi-user auth, cost/quota routing — lower interview
value / high effort relative to time left. Documented as an intentional
scope decision if asked in interviews.

## Conventions / Rules to Follow
- User is new to practical implementation and wants every step spelled out
  explicitly: exact terminal commands, exact PyCharm click-paths, one step
  at a time, waiting for "done" before proceeding to the next step. Don't
  assume familiarity with IDE shortcuts or terminal basics.
- Explain *what* each piece of code does and *why*, in small incremental
  steps, not large code dumps.
- User pastes/edits code themselves after explanation; debugging is
  collaborative — read actual tracebacks/output together, don't guess.
- Prefer free/local solutions given budget constraints.
- Keep `requirements.txt` in sync with every new dependency; hand-curate,
  verify packages are actually installed via `pip show` before adding lines.
- Streamlit apps run via `streamlit run <file>` in terminal — never PyCharm's
  Run button.
- Claude has no direct filesystem access to the user's Mac — all file edits
  are given as exact paste-able content for the user to apply themselves in
  PyCharm; user confirms with "done" after each edit before proceeding.
- Verify claims manually (claim-by-claim, reading actual output/context)
  rather than accepting aggregate scores or theories at face value — this
  pattern has repeatedly caught real issues (RAGAS faithfulness
  investigation, the EnsembleRetriever weight-vs-k misunderstanding, the
  async-ingestion thread-switch bug this session).
- Before committing, check `git status`/`git diff` per modified file rather
  than assuming only intended files changed.
- Given the 20-day placement-prep timeline, favor pragmatic/simple
  implementations defensible in an interview over premature complexity.
- **User is ahead of the original 20-day deadline and is now doing optional
  polish work** (streaming, tests, README, etc.) — these are bonus items,
  not catching up on missed scope. All 5 originally-planned stretch features
  were genuinely complete before this session began.
- When told to proceed through a batch of items "one by one" or "all of the
  above," do not stop to ask which one's next between items — proceed
  through the full sequence, only pausing for "done" confirmation on
  individual edits, not for direction-choosing between items.
- **Final priority list is fixed and user will say "next" to advance through
  it** — Claude should follow the list in "Open Issues" above in order,
  without re-asking which item is next, when the user says "next."
- Project knowledge / attached snapshot docs can go stale — always
  cross-check against the actual pasted/current file contents rather than
  trusting an attached summary at face value. This includes distinguishing
  Rohit's real project files from unrelated reference/tutorial files that
  may appear in the same context (see "Reference material clarification"
  above).

## Continue from here
All 5 originally-planned stretch features (hybrid search, LangSmith tracing,
LLM-swap abstraction, query caching, async ingestion) are complete,
committed, and were re-verified locally this session — including finding
and fixing a real bug in async ingestion's thread-switch handling (Bug 20).

Rohit is ahead of his original deadline and is now working through an
**optional bonus priority list** (see "Open Issues" above), in order,
advancing one item at a time via "next." **Next session should resume
directly at item 1: implementing token streaming in `streamlit_frontend.py`**
— the diagnostic work and full implementation plan are already done (see
"Streaming investigation" above), so this should move straight to applying
the actual code changes, step by step, starting with the
`extract_stream_chunk_text()` helper.

Local `streamlit_frontend.py` and `rag_utils.py` currently have the Bug 20
fix applied and tested, but **not yet pushed to GitHub or redeployed** —
this stays queued for the deployment phase (list item 7), per Rohit's
explicit instruction to do all deployment-related work last.
```

That's the complete file, written from the top — not a patch. Paste it in over your existing `PROJECT_NOTES.md`, save, and it'll carry full context (today's bug fix, verification results, streaming diagnostic findings, and the exact priority list) into any future session without you re-explaining anything.