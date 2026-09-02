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
- Testing: `pytest` (see "Automated Tests" below)

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
  Unused `tools_condition` import removed (cosmetic cleanup, custom
  `route_after_chat`/`route_after_guardrail` were already doing the real
  routing).
  **This session**: `is_query_appropriate()`'s internal YES/NO classifier
  call now tags itself via `llm.invoke(check_prompt, config={"tags":
  ["guardrail_classifier"]})` — needed so the Streamlit streaming layer can
  reliably exclude this internal call from what gets shown to the user (see
  "Streamlit streaming bugs fixed" below). Explanatory one-line comments
  added above every function in this file (readability pass, no behavior
  change beyond the tag).
- **`streamlit_frontend.py`** — sidebar: New Chat button, PDF uploader +
  "Process PDF" button, live polling of `get_ingestion_status`, thread list
  with switch-and-restore. Main area renders history, then either an
  Approve/Reject card or `st.chat_input`. Normal chat turns stream via
  `chatbot.stream()` + `st.write_stream()`; the Approve/Reject resume path
  deliberately still uses plain `.invoke()` (explicit scope decision, not a
  bug). PDF-replace confirmation: uploading a second PDF to a thread that
  already has one loaded requires an explicit "Yes, replace it" click
  instead of silently overwriting. `cleanup_finished_ingestions()` sweeps
  temp files for any thread whose ingestion has actually finished,
  regardless of which thread is active.
  **This session — two real bugs found and fixed** (see "Streamlit
  streaming bugs fixed" below for full detail):
  1. Removed a dead code block gated on `"ingesting_temp_path" in
     st.session_state` (singular, old flag) — nothing sets that key
     anymore since the per-thread `ingesting_temp_paths` (plural) dict
     replaced it in an earlier session; the block was unreachable and just
     duplicated logic already handled correctly elsewhere.
  2. Fixed `stream_chat_turn()` so a guardrail refusal actually streams to
     the user instead of showing a blank assistant bubble, **and** so the
     guardrail's internal YES/NO classifier text doesn't leak into that
     same bubble (was showing as `"NOI'm not able to help..."`). Fixed via
     tag-based filtering (`"guardrail_classifier" not in
     metadata.get("tags", [])`), not text-based filtering — matches the
     project's existing "explicit flags over fragile string-matching"
     principle (same reasoning as the earlier `blocked: bool` fix).
- **`rag_utils.py`** — `load_and_split_pdf()`, `build_vector_store()`
  (FAISS), `build_bm25_retriever()`, `get_retriever()` (hybrid
  `EnsembleRetriever`, FAISS `k=4` + BM25 `bm25_k=2`, `weights=[0.7, 0.3]`).
  Per-thread state: `_THREAD_RETRIEVERS`, `_THREAD_METADATA` (in-memory,
  resets on restart). `_THREAD_RETRIEVERS[thread_id] = ...` overwrites on
  re-ingestion by design (see "Key Decisions" below).
  `ingest_pdf_for_thread()` (sync, used by `evaluation.py`),
  `_ingest_pdf_worker()` + `start_ingestion_async()` + `get_ingestion_status()`
  (async, threaded), `get_retriever_for_thread()`. `_QUERY_CACHE` dict +
  `_clear_thread_cache()` (invalidates cache on re-ingestion). `rag_tool` —
  checks cache first, else retrieves, caches only successful results, returns
  `cache_hit: true/false`.
- **`evaluation.py`** — standalone RAGAS harness. Imports consolidated to
  the top of the file (no behavior change).
- **`requirements.txt`** — hand-curated. Notable: `langchain-classic`,
  `rank_bm25`, `ragas==0.3.9` (pinned), `langchain-google-vertexai`,
  `pytest` (unpinned).
- **`test_calculator.py`, `test_guardrail.py`, `test_rag_caching.py`** —
  see "Automated Tests" below.
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
  YES/NO) only when the last message is a `HumanMessage`. That LLM call is
  now tagged `"guardrail_classifier"` (see this session's fix above) so it
  can be reliably distinguished from real assistant output downstream.
- `chat_node` builds a per-invocation `SystemMessage` injecting `thread_id`
  so the model knows what to pass to `rag_tool`; also instructs multi-step
  sub-question breakdown, explicit "doesn't cover this topic" fallback, and
  a hard rule against naming methods/concepts not present in retrieved
  context (groundedness guardrail).
- Only `search_tool` is gated behind human approval — sole tool making an
  external/untrusted network call.

## Streaming (implemented and verified)

**Why it was needed**: the app previously used blocking `chatbot.invoke()`
for every turn — no visible activity until the full answer was ready.
Streaming was deliberately dropped early on when HITL/`interrupt()` was
first added (to avoid two hard problems at once) and re-added in a later
session.

**What was built**:
1. `extract_stream_chunk_text(content)` — a per-chunk text extractor
   (mirrors `extract_text()`, but joins fragments with `""` instead of `" "`,
   since streamed chunks are partial word fragments, not whole blocks).
2. `stream_chat_turn(user_input, config, accumulated_holder)` — a generator
   fed into `st.write_stream()`. Iterates `chatbot.stream(..., stream_mode="messages")`,
   filters to `metadata['langgraph_node']` in `("chat_node", "guardrail_node")`
   **and** excludes anything tagged `"guardrail_classifier"` (this session's
   fix — see below), and accumulates the full text into
   `accumulated_holder[0]` (a one-item list, needed because a generator
   can't reassign an enclosing-scope variable directly — mutating a list
   sidesteps that).
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

## Streamlit streaming bugs fixed (this session)

Two real bugs found during a code review pass and fixed, in order:

**Bug 1 — dead code, no user-visible symptom.** A leftover block gated on
`if "ingesting_temp_path" in st.session_state:` (singular) still existed
near the bottom of the sidebar section, duplicating the async-ingestion
polling logic that already runs correctly just above it (keyed off the
current, per-thread `ingesting_temp_paths` dict). Nothing in the codebase
sets the singular key anymore, so the block was unreachable — deleted
outright, no behavior change.

**Bug 2 — guardrail refusal invisible, then garbled, now fixed.**
`stream_chat_turn()` originally filtered the stream to only
`metadata['langgraph_node'] == "chat_node"`. Since the guardrail's refusal
`AIMessage` is emitted from `guardrail_node`, not `chat_node`, it was
filtered out entirely — a blocked query produced a **blank assistant
bubble** in the UI (verified via screenshot before fixing).

First fix attempt — widen the node filter to `("chat_node",
"guardrail_node")` — surfaced a second, subtler bug: `guardrail_node`'s
internal YES/NO classifier call (`is_query_appropriate()`) also runs
*inside* `guardrail_node`'s execution and therefore also gets tagged
`langgraph_node="guardrail_node"` in the stream. This caused the classifier's
own `"NO"` token to leak into the same bubble as the real refusal message,
rendering as `"NOI'm not able to help with that request..."` (verified via
screenshot).

**Final fix**: tagged the classifier's own LLM call with
`config={"tags": ["guardrail_classifier"]}` in `langgraph_backend.py`, then
added `if "guardrail_classifier" in metadata.get("tags", []): continue` to
`stream_chat_turn()`'s filter. This is tag-based exclusion, not text-based
— deliberately chosen so a legitimate assistant answer that happens to
start with the word "No" is never mistakenly filtered (matches the
project's existing principle of explicit flags over fragile string-matching,
same reasoning as the earlier `blocked: bool` guardrail-routing fix).

**Verified via two screenshots**: (1) blank-bubble bug confirmed present
before the first fix; (2) after the tag fix, the refusal bubble reads
cleanly — "I'm not able to help with that request. Please ask a different
question." — with no leading "NO". Happy-path re-verification (a normal,
allowed question still streams correctly after this change) was flagged as
worth confirming but not yet explicitly re-tested/screenshotted — see "Open
Issues" below.

## Hallucination detection — discussed, scope decision made (this session)

**Context**: Rohit shared an external post describing three production
hallucination-mitigation techniques — self-consistency (multi-sampling),
retrieval grounding via a dedicated NLI model, and LLM-as-a-judge — and
asked whether this project implements any of them.

**Honest assessment given, mapped against the actual codebase**:
- **Self-consistency (multi-sampling)**: not implemented. `chat_node` calls
  the LLM once per turn; nothing samples multiple times and checks for
  contradiction.
- **Retrieval grounding via NLI**: not implemented as a runtime checker.
  What exists instead: (a) prompt-based grounding instructions in
  `chat_node`'s system prompt (soft, self-policing, unverified), and (b)
  the RAGAS `Faithfulness()` metric in `evaluation.py`, which is
  conceptually similar (breaks the answer into claims, checks each against
  retrieved context via an LLM judge) but runs **offline**, manually, on a
  fixed 3-question test set — never live, never blocking a real answer.
- **LLM-as-a-judge**: not implemented at runtime. `guardrail_node` is a
  different mechanism — it screens the incoming *question* before the LLM
  answers, not the outgoing *answer* for unsupported claims. RAGAS's
  judge-based metrics are, again, the offline equivalent.

**Decision: leave as-is, no code changes.** Reasoning discussed and
agreed:
- The existing RAGAS faithfulness investigation (documented in detail
  under "RAGAS Baselines" / earlier "Retrieval architecture" debugging
  trail from prior sessions) is already a strong, defensible interview
  story on exactly this topic — arguably more impressive than a bolted-on
  runtime checker, since it demonstrates actually investigating and
  root-causing a metric rather than trusting it blindly.
- A real-time faithfulness/NLI gate would mean a second LLM call on every
  single answer, which is a real quota risk on the free-tier Gemini setup
  that has already hit 429s once this project.
- Remaining time is better spent on the still-open priority items
  (README, resume writeup, deployment) than a new feature this late in
  the timeline.

**What to say if asked in an interview** (agreed phrasing, paraphrased):
"I don't gate live answers with a runtime verifier — that's a known gap.
What I do have is prompt-level grounding instructions telling the model to
say 'not in the document' rather than guess, plus an offline RAGAS
faithfulness evaluation harness I use to measure and debug groundedness
issues after the fact. If I were taking this to production, the next step
would be a lightweight NLI or LLM-judge check on the final answer before
it's shown to the user." This distinguishes "I understand the technique
and where mine falls short" from not knowing the concept — the actual
thing being screened for.

## PDF-replace confirmation (implemented and verified)

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

## Automated Tests

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

## Bug fixed: SSL certificate verification error

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
- **Single-PDF-per-thread, with explicit replace confirmation**: pragmatic
  scope boundary for a placement-prep timeline; documented and defensible
  in an interview, and now has a small UX safeguard so it can't look like
  a bug in a live demo.
- **Streaming: chat turns streamed, resume path not streamed**: explicit,
  explainable scope boundary — not an oversight.
- **No runtime hallucination gate (self-consistency / NLI / LLM-judge)**
  (decided this session): offline RAGAS faithfulness evaluation plus
  prompt-level grounding instructions are the accepted current state;
  adding a live runtime verifier was explicitly assessed and deferred —
  real quota risk (second LLM call per answer on an already-429-prone free
  tier) outweighs benefit this late in the timeline. Documented above
  under "Hallucination detection" with the exact interview framing to use.
- **Weather/stock/other new RAG tools — explicitly deferred, not planned
  soon.** Discussed and deliberately declined for now: doesn't strengthen
  the *RAG* story (it's tool-calling, not retrieval-augmented generation),
  adds real scope (API keys, HITL-gating for the new external call,
  prompt-engineering for tool selection), and README/resume/deployment
  matter more for interview-readiness than tool breadth. Revisit only if
  time remains after the current priority list is fully done.
- **No git commits between individual work steps** — Rohit's standing
  instruction: batch changes locally and only commit/push around natural
  checkpoints, not after every small item. (This session was a documented
  exception — see Git section below.)

## Bugs Fixed (cumulative, latest first)
22. **Guardrail refusal not streaming, then garbled by leaked classifier
    text** — see "Streamlit streaming bugs fixed" above. Fixed via
    widening the stream's node filter plus a `"guardrail_classifier"` tag
    on the internal YES/NO check, so it can be excluded without relying on
    fragile text matching.
21. **SSL certificate verification error blocking `duckduckgo_search`** —
    see "Bug fixed" above. Fixed via `Install Certificates.command`.
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
- **This session's commits**:
  - `ea5b42c` — "Add explanatory comments across backend/eval/rag_utils
    files; add unit tests for calculator, guardrail, and rag caching" (8
    files: `evaluation.py`, `langgraph_backend.py`, `llm_config.py`,
    `rag_utils.py`, `streamlit_frontend.py`, `test_calculator.py`,
    `test_guardrail.py`, `test_rag_caching.py`).
  - `780628d` — "Fix guardrail refusal not streaming; tag-filter internal
    YES/NO classifier call to keep it out of chat output" (2 files:
    `langgraph_backend.py`, `streamlit_frontend.py`).
  - Pushed successfully: `cf7a0c3..ea5b42c` then `ea5b42c..780628d`, both
    on `main -> main`. Repo fully synced as of end of this session.
- `files.zip` sitting untracked in the project root — confirmed by Rohit
  it's not part of the project and can be ignored; not added to
  `.gitignore`, just left untracked for now.
- **Prior session's commit**: `8e1d828` — "Add token streaming, pytest
  suite, PDF-replace confirmation, cleanup" (10 files).
- **Merge commit** (prior session): `159c6e7` — merged in a remote-only
  commit (`9212dd5`, "Added Dev Container Folder") that existed on GitHub
  but not in the local clone. No conflicts (unrelated `.devcontainer`
  file).
- **Staging quirk resolved (prior session)**: `test_cache.py` and
  `test_streaming.py` (throwaway diagnostic scripts) had been auto-staged
  by PyCharm as "new files" despite being deleted from disk. Fixed via
  `git rm --cached test_cache.py test_streaming.py` before committing —
  neither file is tracked or present anymore.
- **`.gitignore` bug fixed (prior session)**: the entry `PROJECT KNOWLEDGE`
  (no extension) didn't actually match the real file
  `PROJECT KNOWLEDGE.txt`, so it was showing as untracked instead of
  ignored. Fixed by changing the pattern to `PROJECT KNOWLEDGE*`.
- **Convention (explicit, from Rohit)**: no git commits/pushes between
  individual work items going forward — batch locally, commit/push around
  natural checkpoints. (Both prior and this session included explicit
  exceptions Rohit requested.)
- `.gitignore` covers: `.venv/`, `.env`, `__pycache__/`, `*.pyc`, `*.db`,
  `*.db-shm`, `*.db-wal`, `temp_*.pdf`, `evaluation_results.json`,
  `test.pdf`, `.idea/`, `.DS_Store`, `requirements_current.txt`,
  `.agents/`, `.claude/`, `PROJECT KNOWLEDGE*`.

## Open Issues / Priority List

1. ~~Streaming + tool-status badge~~ — **DONE, verified.** Tool-status
   badge deliberately not built (optional future idea, not current scope).
2. ~~Automated tests (pytest)~~ — **DONE**, 15 tests passing.
3. ~~Single-PDF-per-thread behavior~~ — **DONE**, decided (option 2:
   confirm-before-replace) and implemented/verified.
4. ~~Streamlit streaming bugs (dead code, guardrail refusal not
   showing/garbled)~~ — **DONE this session**, both fixed and verified via
   screenshots.
5. **Minor: happy-path re-verification after the guardrail-streaming
   fix.** A normal, allowed question streaming correctly was flagged as
   worth confirming after the tag-filter change went in, but not yet
   explicitly re-tested/screenshotted. Quick sanity check, not expected to
   reveal anything — do this before or during the next session.
6. **README.md for the GitHub repo — NEXT major item.** Not started.
   Still need to decide: audience (recruiter-skim vs. technical-detailed
   vs. both in sections) and format (draft in chat first vs. paste-ready
   `.md` directly).
7. **Resume/portfolio writeup** — not started.
8. **Deployment (Days 19–20) — deliberately last.** Redeploy to Streamlit
   Cloud with all recent changes (now pushed to GitHub), then re-verify
   all 5 stretch features + streaming + PDF-replace confirmation on the
   live deployed app.

**Explicitly deferred, not gaps**: semantic chunking, multi-user auth,
cost/quota routing, multi-PDF-per-thread support, new tool integrations
(weather/stocks/etc.), and a runtime hallucination-detection gate
(self-consistency / NLI / LLM-as-judge) — all discussed and deliberately
scoped out as lower interview value / high effort / real risk relative to
time and quota left. Documented as intentional scope decisions, each with
its own defensible interview framing (see "Key Decisions" and
"Hallucination detection" above).

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
- Verify claims manually (claim-by-claim, real output/context, and — this
  session — real screenshots of before/after UI behavior) rather than
  accepting aggregate scores/theories at face value.
- Check `git status`/`git diff` per modified file before committing —
  never assume only intended files changed. If something unexpected shows
  up (e.g. an unexplained modified file, or an untracked file like
  `files.zip`), confirm with Rohit what it is before including or
  excluding it from a commit.
- **No git commits between individual work steps** — batch locally, only
  commit/push around natural checkpoints (see Git section above), unless
  Rohit explicitly requests an exception.
- When told to proceed through a batch "one by one" or "all of the above,"
  don't stop to ask which one's next between items — proceed through the
  full sequence, only pausing for "done" confirmation on individual edits.
- **Be honest and conservative about scope, especially anything Rohit
  needs to be able to defend in an interview.** Rohit has explicitly said
  he doesn't want to add anything "hyped" or that he can't fully explain
  if asked — when proposing a new feature, explicitly weigh
  interview-defensibility, not just technical interest. When in doubt,
  favor the smaller/simpler/more explainable option. This applies to
  declining features too (see "Hallucination detection" this session) —
  when the honest answer is "I'm aware of this technique but haven't
  implemented it," give Rohit that framing directly rather than either
  overstating current capability or silently adding scope he didn't ask
  for.
- Throwaway diagnostic/test scripts (e.g. `test_cache.py`,
  `test_streaming.py`) should be deleted after use, and `git status`
  checked before the next commit to catch any accidental auto-staging by
  PyCharm (a recurring quirk).
- Project knowledge / attached snapshot docs can go stale — cross-check
  against actual pasted/current file contents rather than trusting an
  attached summary at face value.

## Continue from here
This session: added explanatory one-line comments across
`evaluation.py`, `langgraph_backend.py`, `llm_config.py`, `rag_utils.py`,
`streamlit_frontend.py`, plus three new pytest files
(`test_calculator.py`, `test_guardrail.py`, `test_rag_caching.py`, 15
tests total, all passing). During that review, found and fixed two real
bugs in `streamlit_frontend.py`'s streaming path (dead code block; a
guardrail refusal that was invisible, then garbled by leaked classifier
text once first widened). Both committed and pushed (`ea5b42c`,
`780628d`). Also discussed the "self-consistency / NLI / LLM-as-judge"
hallucination-detection techniques against the actual codebase and made an
explicit, documented decision not to add a runtime gate right now (see
"Hallucination detection" section above for the full reasoning and the
interview-ready explanation to give if asked).

**Next session should**: (1) do the quick happy-path streaming
re-verification flagged in Open Issues item 5, then (2) move to README.md
(item 6) — two open questions to resolve first: audience (recruiter-skim
vs. technical vs. both) and format (chat draft first vs. paste-ready file
directly). After README: resume/portfolio writeup (item 7), then
deployment (item 8) last.

Rohit was explicit that new RAG tools (weather, stocks, etc.) are **not**
wanted right now, and that a runtime hallucination-detection layer is a
deliberate, documented "not now" rather than an oversight — don't propose
either again unless Rohit brings it up.
