# RAG Chatbot — Project Notes

Project path: `/Users/hawkeyez007/Desktop/RAG_Chat_Bot`
Goal: Resume-ready RAG chatbot for placement interviews. Python 3.14, PyCharm,
Streamlit frontend, LangGraph backend, Gemini free tier, local embeddings.
GitHub: github.com/d4t4forge-debugX/RAG_Chat_Bot
Deployment: not yet done (Streamlit Community Cloud, planned last).

## Stack
- LLM: Google Gemini via `llm_config.py`'s `get_llm()` factory — defaults to
  `gemini-3.5-flash-lite` / `thinking_level="low"`, overridable via
  `LLM_MODEL` / `LLM_THINKING_LEVEL` env vars (`ChatGoogleGenerativeAI`)
- Embeddings: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS (`faiss-cpu`)
- Lexical retrieval: BM25 (`rank_bm25`, via `langchain_community.retrievers.BM25Retriever`)
- Hybrid retrieval: `EnsembleRetriever` from `langchain_classic.retrievers`
  (moved out of core `langchain` in LangChain v1.0 — `langchain-classic` is a
  direct dependency because of this)
- Persistence: `SqliteSaver` (`langgraph-checkpoint-sqlite`), db file `chatbot.db`
- Web search: `ddgs` package (renamed from `duckduckgo-search`)
- Evaluation: `ragas==0.3.9` (pinned — see manual patch note below), plus
  `langchain-google-vertexai` (needed for the patch's fallback import)
- Observability: LangSmith tracing via `.env` vars only
- Caching: in-memory dict cache on `rag_tool`, keyed by `(thread_id, query)`
- Async ingestion: background `threading.Thread`-based PDF ingestion with
  live progress polling from the Streamlit UI
- Testing: `pytest` (added this session — see "Automated Tests" below)

## Files (roles)
- **`llm_config.py`** — single source of truth for LLM selection. `get_llm()`
  reads `LLM_MODEL`/`LLM_THINKING_LEVEL` from env, falls back to
  `gemini-3.5-flash-lite`/`low`. Used by `langgraph_backend.py` and
  `evaluation.py`.
- **`langgraph_backend.py`** — graph: `guardrail_node` → `chat_node` ⇄ `tools`,
  with `human_review_node` gating only `duckduckgo_search`. `ChatState` =
  `{messages: Annotated[list[BaseMessage], add_messages], blocked: bool}`.
  Tools bound: `search_tool` (DuckDuckGo), `calculator`, `rag_tool`.
  `checkpointer` = `SqliteSaver` on `chatbot.db`. `retrieve_all_threads()`
  helper. `__main__` block runs a terminal HITL smoke test.
  Unused `tools_condition` import **removed this session** (cosmetic cleanup,
  custom `route_after_chat`/`route_after_guardrail` were already doing the
  real routing).
- **`streamlit_frontend.py`** — sidebar: New Chat button, PDF uploader +
  "Process PDF" button, live polling of `get_ingestion_status`, thread list
  with switch-and-restore. Main area renders history, then either an
  Approve/Reject card or `st.chat_input`.
  **Streaming implemented this session** (see below) — normal chat turns now
  stream via `chatbot.stream()` + `st.write_stream()`; the Approve/Reject
  resume path deliberately still uses plain `.invoke()` (explicit scope
  decision, not a bug).
  **PDF-replace confirmation added this session** (see below) — uploading a
  second PDF to a thread that already has one loaded now requires an
  explicit "Yes, replace it" click instead of silently overwriting.
  `cleanup_finished_ingestions()` sweeps temp files for any thread whose
  ingestion has actually finished, regardless of which thread is active.
- **`rag_utils.py`** — `load_and_split_pdf()`, `build_vector_store()`
  (FAISS), `build_bm25_retriever()`, `get_retriever()` (hybrid
  `EnsembleRetriever`, FAISS `k=4` + BM25 `bm25_k=2`, `weights=[0.7, 0.3]`).
  Per-thread state: `_THREAD_RETRIEVERS`, `_THREAD_METADATA` (in-memory,
  resets on restart). `_THREAD_RETRIEVERS[thread_id] = ...` **overwrites on
  re-ingestion by design** — this was investigated and decided this session
  (see "Key Decisions" below), not left ambiguous anymore.
  `ingest_pdf_for_thread()` (sync, used by `evaluation.py`),
  `_ingest_pdf_worker()` + `start_ingestion_async()` + `get_ingestion_status()`
  (async, threaded), `get_retriever_for_thread()`. `_QUERY_CACHE` dict +
  `_clear_thread_cache()` (invalidates cache on re-ingestion). `rag_tool` —
  checks cache first, else retrieves, caches only successful results, returns
  `cache_hit: true/false`.
- **`evaluation.py`** — standalone RAGAS harness. Imports were reorganized
  this session (consolidated to the top of the file, no behavior change —
  the file previously had imports scattered mid-file from an old
  copy-paste pattern).
- **`requirements.txt`** — hand-curated. Notable: `langchain-classic`,
  `rank_bm25`, `ragas==0.3.9` (pinned), `langchain-google-vertexai`,
  `pytest` (added this session, unpinned).
- **`test_calculator.py`, `test_guardrail.py`, `test_rag_caching.py`** —
  new this session, see "Automated Tests" below.
- `test.pdf` — Hands-On Machine Learning by Aurélien Géron (gitignored, not
  committed — copyrighted).

## Architecture (graph shape)

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

- `guardrail_node` runs `is_query_appropriate()` (single LLM call, expects
  YES/NO) only when the last message is a `HumanMessage`.
- `chat_node` builds a per-invocation `SystemMessage` injecting `thread_id`
  so the model knows what to pass to `rag_tool`; also instructs multi-step
  sub-question breakdown, explicit "doesn't cover this topic" fallback, and
  a hard rule against naming methods/concepts not present in retrieved
  context (groundedness guardrail).
- Only `search_tool` is gated behind human approval — sole tool making an
  external/untrusted network call.

## Streaming (implemented and verified this session)

**Why it was needed**: the app previously used blocking `chatbot.invoke()`
for every turn — no visible activity until the full answer was ready.
Streaming was deliberately dropped early on when HITL/`interrupt()` was
first added (to avoid two hard problems at once) and never re-added until
now.

**What was built**:
1. `extract_stream_chunk_text(content)` — a per-chunk text extractor
   (mirrors `extract_text()`, but joins fragments with `""` instead of `" "`,
   since streamed chunks are partial word fragments, not whole blocks).
2. `stream_chat_turn(user_input, config, accumulated_holder)` — a generator
   fed into `st.write_stream()`. Iterates `chatbot.stream(..., stream_mode="messages")`,
   filters to only `metadata['langgraph_node'] == "chat_node"` chunks
   (discarding the guardrail's internal YES/NO check, which also streams),
   and accumulates the full text into `accumulated_holder[0]` (a one-item
   list, needed because a generator can't reassign an enclosing-scope
   variable directly — mutating a list sidesteps that).
3. After the stream loop ends, `chatbot.get_state(config)` is called and
   `state.tasks` is checked for `task.interrupts` — the same pattern
   `load_conversation()` already used. Interrupts never appear *inside* the
   stream itself; they can only be detected by checking state afterward.
4. **Scope decision (deliberate, unchanged)**: the Approve/Reject resume
   path still uses plain `chatbot.invoke()`, not streamed. Reasonable
   boundary — resumed responses are typically short.
5. **Tool-status badge ("🔧 Using `tool_name`...") — deliberately NOT built.**
   Flagged as a possible future follow-up, not part of current scope.

**Verified in browser**: sent a 500-word test prompt, confirmed via a
temporary debug print that text arrives in ~19 real chunks (~100–160 chars
each, roughly sentence-sized) — genuine incremental streaming, just fast
enough that it can look like a near-instant flash at full speed. Debug
prints and an artificial `time.sleep()` used only for diagnosis were removed
afterward. Also verified the interrupt path end-to-end: streaming a
search-triggering query still correctly shows the Approve/Reject card after
the stream ends.

## PDF-replace confirmation (implemented and verified this session)

**The question** (previously open): `_THREAD_RETRIEVERS[thread_id] = retriever`
overwrites on each ingestion — uploading a second PDF to a thread silently
replaces the first one's retriever. Investigated three options: (1) leave
as-is and just document it, (2) leave the overwrite behavior but add an
explicit UI confirmation before it happens, (3) support multiple PDFs per
thread (significant new complexity — new retrieval logic, cache-key
implications, UI to show multiple loaded docs).

**Decision: option 2.** Rationale: option 1 leaves a rough edge that could
look like a bug in a live interview demo; option 3 is real scope creep for
a placement-prep timeline. Option 2 is a small, safe, easily-explained
addition ("I noticed a silent-replace could confuse a user in a demo, so I
added an explicit confirmation step") — genuinely strengthens the interview
story rather than just excusing a limitation.

**What was built**: a `session_state["confirm_pdf_replace"]` boolean flag.
If no document is loaded yet for the current thread, "Process PDF" behaves
exactly as before (no confirmation needed). If a document is already
loaded and not yet confirmed, the first "Process PDF" click sets the flag
and shows a warning ("This will replace the document currently loaded for
this chat.") plus "Yes, replace it" / "Cancel" buttons — ingestion doesn't
start until "Yes, replace it" is explicitly clicked. The actual
ingestion-kickoff logic was pulled into one shared helper,
`_start_pdf_ingestion()`, used by both the no-existing-doc path and the
confirmed-replace path, to avoid duplicating those steps.

**Verified in browser**: first upload processes with no warning; second
upload to the same thread shows the warning; "Cancel" leaves the original
document intact; "Yes, replace it" proceeds and successfully re-indexes.

## Automated Tests (added this session)

`pytest` installed and added to `requirements.txt` (unpinned). 15 tests
across 3 files, all passing:

- **`test_calculator.py`** (6 tests) — all four operations (add/sub/mul/div)
  plus both explicit error paths (division by zero, unsupported operation).
  Calls `calculator.invoke({...})` since it's a `@tool`-decorated function.
- **`test_guardrail.py`** (4 tests) — tests `is_query_appropriate()`'s
  YES/NO parsing logic in isolation, **mocking** `langgraph_backend.llm`
  (patched where it's *used*, not where it's defined — a common mocking
  gotcha) so no real API calls are made. Covers: uppercase YES → True,
  NO → False, lowercase/mixed-case "yes, this is fine" → True, and a
  list-of-content-blocks response shape (matching how Gemini sometimes
  actually responds) → still parsed correctly.
- **`test_rag_caching.py`** (5 tests) — tests `rag_tool`'s caching logic
  by mocking `rag_utils.get_retriever_for_thread` to return a fake
  retriever (no real PDF/embedding needed). Uses an `autouse=True` pytest
  fixture to clear the module-level `_QUERY_CACHE` dict before/after each
  test (since it's shared global state across tests). Covers: first call is
  a cache miss, identical second call is a cache hit (retriever only
  invoked once), cache key is case/whitespace-insensitive
  (`query.strip().lower()`), different threads don't share a cache entry,
  and `_clear_thread_cache(thread_id)` only clears that specific thread's
  entries (the mechanism `ingest_pdf_for_thread()` calls internally on
  re-ingestion).

Running `pytest -v` (whole suite) confirmed no cross-file interference.
Not currently wired into a pre-commit hook or CI — run manually before
commits at Rohit's discretion.

## Bug fixed this session: SSL certificate verification error

**Symptom**: `duckduckgo_search` failed on every attempt with
`ddgs.exceptions.DDGSException: ... [SSL: CERTIFICATE_VERIFY_FAILED] self-signed
certificate in certificate chain`. This caused a *secondary* symptom that
looked like an infinite loop: the system prompt instructs the model to
rephrase and retry when a search doesn't return useful results, so it kept
retrying (each retry re-triggering the Approve/Reject gate) since the
search failed identically every time regardless of query rephrasing.

**Root cause**: a well-known macOS Python issue — Python's bundled
`certifi` CA bundle isn't linked to the system trust store by default,
so any HTTPS request fails cert verification.

**Fix**: ran `/Applications/Python\ 3.14/Install\ Certificates.command`
(the official installer's bundled fix script), which upgrades `certifi`
and re-links the certificate bundle. Verified fixed: a follow-up search +
approve completed successfully (with 2–3 legitimate rephrase-and-retry
cycles before landing on a real answer — confirmed as correct multi-step
behavior per the system prompt, not a bug).

## Key Decisions & Why
- **Local embeddings over OpenAI**: no per-call cost, strong interview
  talking point.
- **Gemini free tier + `thinking_level="low"`**: budget constraint.
- **Per-thread RAG state kept in-memory**: simple, per-conversation PDF
  isolation; resets on restart — accepted, documented limitation.
- **HITL gates only web search**: sole tool touching an untrusted external
  source; `calculator`/`rag_tool` are deterministic/internal.
- **Guardrails are prompt-based, not separate classifiers**: simpler to
  build and explain than a dedicated groundedness model.
- **Hybrid retrieval tuning (FAISS k=4, BM25 bm25_k=2, weights=[0.7,0.3])**:
  `EnsembleRetriever` merges the *union* and re-ranks by weight — doesn't
  drop candidates. The actual fix for BM25 noise was shrinking `bm25_k` at
  the source, not reweighting.
- **Caching only retrieval, not final LLM output**: avoids staleness
  questions disproportionate to benefit.
- **Single-PDF-per-thread, with explicit replace confirmation** (decided
  this session): pragmatic scope boundary for a placement-prep timeline;
  documented and defensible in an interview, and now has a small UX
  safeguard so it can't look like a bug in a live demo.
- **Streaming: chat turns streamed, resume path not streamed** (decided
  this session): explicit, explainable scope boundary — not an oversight.
- **Weather/stock/other new RAG tools — explicitly deferred, not planned
  soon.** Discussed and deliberately declined for now: doesn't strengthen
  the *RAG* story (it's tool-calling, not retrieval-augmented generation),
  adds real scope (API keys, HITL-gating for the new external call,
  prompt-engineering for tool selection), and README/resume/deployment
  matter more for interview-readiness than tool breadth. Revisit only if
  time remains after the current priority list is fully done.
- **No git commits between individual work steps** — Rohit's explicit
  instruction: batch changes locally and only commit/push around
  deployment time, not after every small item.

## Bugs Fixed (cumulative, latest first)
21. **SSL certificate verification error blocking `duckduckgo_search`** —
    see "Bug fixed this session" above. Fixed via
    `Install Certificates.command`.
20. Async ingestion progress lost on thread switch — root cause was a
    single session-wide flag instead of a per-thread dict; fixed with
    `st.session_state["ingesting_temp_paths"]` (per-thread) and
    `cleanup_finished_ingestions()` (only cleans up threads whose job is
    actually done/errored, never based on navigation alone). Verified via
    exact repro test.
19. LangSmith tracing, LLM-swap, caching, async ingestion all worked
    cleanly on first implementation. One git staging quirk (PyCharm
    auto-staged a throwaway `test_cache.py` before deletion) — resolved.
18–1. See prior sessions: message/messages typo, tool-definition-order bug,
    `duckduckgo-search`→`ddgs` rename, Unicode surrogate crash in
    embeddings, Gemini model deprecations, free-tier 429s, `ragas` import
    crash (patch below), HITL retry-after-reject bug, Streamlit rerun
    staleness, debug print flood, leftover temp files, dead code, leaked
    API key (rotated), oversized `test.pdf` accidentally committed
    (removed via `git rm --cached`), RAGAS faithfulness anomaly (0.27,
    traced to ungrounded term naming), `ingest_pdf_for_thread` signature
    mismatch, `EnsembleRetriever` moved to `langchain-classic` in
    LangChain v1.0.

## Manual patch required after any fresh venv rebuild
`ragas==0.3.9` unconditionally imports `langchain_community.chat_models.vertexai`,
which no longer exists in current `langchain-community`. Breaks any script
importing `ragas` (i.e. `evaluation.py`) with a `ModuleNotFoundError`.

File to patch: `.venv/lib/python3.14/site-packages/ragas/llms/base.py` —
wrap the `ChatVertexAI`/`VertexAI` imports in try/except, fall back to
`langchain_google_vertexai`, filter `None` out of
`MULTIPLE_COMPLETION_SUPPORTED`. Clear `ragas`'s `__pycache__` after
patching (stale `.pyc` can mask the fix):
```
find .venv -path "*/ragas/**/__pycache__" -exec rm -rf {} +
```
**This is a manual edit to an installed package — cannot be captured in
`requirements.txt`. Must be reapplied after every fresh `pip install`.**

## RAGAS Baselines
**Hybrid search, `bm25_k=2`, `weights=[0.7,0.3]` (FINAL, accepted config)**:
```
Run 1: faithfulness 0.8333, answer_relevancy 0.8116, context_precision 0.5278
Run 2: faithfulness 0.9333, answer_relevancy 0.8756, context_precision 0.5278
```
`context_precision` is stable/reproducible at 0.5278 across runs (retrieval
is deterministic). `faithfulness`/`answer_relevancy` vary run-to-run due to
RAGAS judge noise on a small n=3 eval set — confirmed by two independent
re-runs. (Full historical baselines, including pre-hybrid and the
`bm25_k` tuning investigation, are in earlier session logs if needed.)

## Git / Repo Hygiene
- **This session's commit**: `8e1d828` — "Add token streaming, pytest
  suite, PDF-replace confirmation, cleanup" (10 files: `.gitignore`,
  `PROJECT_NOTES.md`, `evaluation.py`, `langgraph_backend.py`,
  `rag_utils.py`, `requirements.txt`, `streamlit_frontend.py`, plus 3 new
  test files).
- **Merge commit**: `159c6e7` — merged in a remote-only commit
  (`9212dd5`, "Added Dev Container Folder") that existed on GitHub but not
  in the local clone. No conflicts (unrelated `.devcontainer` file).
- **Pushed successfully**: `9212dd5..159c6e7 main -> main`. Repo is fully
  synced as of end of this session.
- **Staging quirk resolved this session**: `test_cache.py` and
  `test_streaming.py` (throwaway diagnostic scripts) had been auto-staged
  by PyCharm as "new files" despite being deleted from disk. Fixed via
  `git rm --cached test_cache.py test_streaming.py` before committing —
  neither file is tracked or present anymore.
- **`.gitignore` bug fixed this session**: the entry `PROJECT KNOWLEDGE`
  (no extension) didn't actually match the real file
  `PROJECT KNOWLEDGE.txt`, so it was showing as untracked instead of
  ignored. Fixed by changing the pattern to `PROJECT KNOWLEDGE*`.
- Also cleaned up this session: a stray duplicate file,
  `PROJECT_NOTES (1).md`, deleted (was an accidental "Save As" duplicate).
- **Convention (explicit, from Rohit)**: no git commits/pushes between
  individual work items going forward — batch locally, commit/push around
  deployment time only. (This session's commit was an exception Rohit
  explicitly requested due to running low on time/tokens.)
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `*.db`,
  `*.db-shm`, `*.db-wal`, `temp_*.pdf`, `evaluation_results.json`,
  `test.pdf`, `.idea/`, `.DS_Store`, `requirements_current.txt`,
  `.agents/`, `.claude/`, `PROJECT KNOWLEDGE*`.

## Open Issues / Priority List

1. ~~Streaming + tool-status badge~~ — **streaming DONE this session,
   verified.** Tool-status badge deliberately not built (optional future
   idea, not current scope).
2. ~~Automated tests (pytest)~~ — **DONE this session**, 15 tests passing.
3. ~~Single-PDF-per-thread behavior~~ — **DONE this session**, decided
   (option 2: confirm-before-replace) and implemented/verified.
4. **README.md for the GitHub repo — NEXT.** Not started. Still need to
   decide: audience (recruiter-skim vs. technical-detailed vs. both in
   sections) and format (draft in chat first vs. paste-ready `.md`
   directly).
5. **Resume/portfolio writeup** — not started.
6. ~~Cosmetic cleanup~~ — **DONE this session** (removed unused
   `tools_condition` import from `langgraph_backend.py`).
7. **Deployment (Days 19–20) — deliberately last.** Redeploy to Streamlit
   Cloud with all of today's changes (now pushed to GitHub), then
   re-verify all 5 stretch features + streaming + the new PDF-replace
   confirmation on the live deployed app.

**Explicitly deferred, not gaps**: semantic chunking, multi-user auth,
cost/quota routing, multi-PDF-per-thread support, and new tool integrations
(weather/stocks/etc.) — all discussed and deliberately scoped out as lower
interview value / high effort relative to time left. Documented as
intentional scope decisions, defensible if asked in interviews.

## Conventions / Rules to Follow
- Rohit is new to practical implementation — spell out every step
  explicitly: exact terminal commands, exact PyCharm click-paths, one step
  at a time, wait for "done" before proceeding. Don't assume familiarity
  with IDE shortcuts, terminal basics, or tools like `vim` (walk through
  save/exit explicitly if a text editor opens unexpectedly, e.g. during a
  git merge).
- Explain *what* and *why* for every code change, in small increments —
  not large code dumps.
- Rohit pastes/edits code himself in PyCharm; debugging is collaborative —
  read actual tracebacks/output together, don't guess. When something looks
  like a bug, add a small temporary diagnostic (debug print, etc.) to
  confirm root cause before proposing a fix, then remove the diagnostic
  once confirmed.
- Prefer free/local solutions given budget constraints.
- Keep `requirements.txt` in sync with every new dependency; hand-curate,
  verify via `pip show` before adding.
- Streamlit apps run via `streamlit run <file>` in terminal — never
  PyCharm's Run button.
- Claude has no filesystem access to Rohit's Mac — all edits given as
  exact paste-able content; Rohit confirms "done" after each edit.
- Verify claims manually (claim-by-claim, real output/context) rather than
  accepting aggregate scores/theories at face value.
- Check `git status`/`git diff` per modified file before committing —
  never assume only intended files changed. If something unexpected shows
  up (e.g. an unexplained modified file), investigate the actual diff
  before including it in a commit.
- **No git commits/pushes between individual work items** — batch locally,
  only commit/push around deployment time (see Git section above).
- When told to proceed through a batch "one by one" or "all of the above,"
  don't stop to ask which one's next between items — proceed through the
  full sequence, only pausing for "done" confirmation on individual edits.
- **Final priority list is fixed; Rohit says "next" to advance through
  it** — follow the "Open Issues" list above in order without re-asking.
- **Be honest and conservative about scope, especially anything Rohit
  needs to be able to defend in an interview.** Rohit has explicitly said
  he doesn't want to add anything "hyped" or that he can't fully explain
  if asked — when proposing a new feature (e.g. PDF-replace confirmation,
  new tools), explicitly weigh interview-defensibility, not just technical
  interest. When in doubt, favor the smaller/simpler/more explainable
  option.
- Throwaway diagnostic/test scripts (e.g. `test_cache.py`,
  `test_streaming.py`) should be deleted after use, and `git status`
  checked before the next commit to catch any accidental auto-staging by
  PyCharm (a recurring quirk).
- Project knowledge / attached snapshot docs can go stale — cross-check
  against actual pasted/current file contents rather than trusting an
  attached summary at face value.

## Continue from here
Items 1, 2, 3, and 6 of the priority list are now done and verified this
session (streaming, automated tests, single-PDF-per-thread decision,
cosmetic cleanup). Today's work is committed (`8e1d828`) and pushed to
GitHub, including a clean merge of an out-of-sync remote commit
(`159c6e7`). The SSL certificate bug (a real, previously-undiagnosed issue
blocking all web search) was also found and fixed this session.

**Next session should resume at item 4: README.md.** Two open questions
to resolve first: (1) should it target recruiters (concise, feature
highlights) or be more technical (architecture, setup instructions), or
both in sections; (2) does Rohit want a draft reviewed in chat first, or
a paste-ready `.md` file directly. After README, move to item 5 (resume/
portfolio writeup), then item 7 (deployment) — per Rohit's standing
instruction, no further git commits until deployment time.

Rohit was explicit this session that new RAG tools (weather, stocks, etc.)
are **not** wanted right now — flagged as a possible "if time remains
after everything else" idea, not part of the active priority list. Don't
propose adding new tools again unless Rohit brings it up.
