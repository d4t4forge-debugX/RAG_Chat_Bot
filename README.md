# RAG Chatbot

A retrieval-augmented generation chatbot that answers questions from an uploaded PDF, with hybrid search, human-in-the-loop approval for web search, source pages shown under answers, conversation persistence, and a measured evaluation. Built as a portfolio project; the design choices and the measured results below are documented, including where something did not help.

It is not deployed. It runs locally (see [Setup](#setup)).

---

## What it does

Upload a PDF, ask questions about it, and get answers grounded in the document, with the PDF pages the answer was retrieved from shown underneath. If the document does not cover something, the assistant can search the web instead, but only after you approve that specific search, so nothing leaves your machine without permission. Retrieval quality is measured, not assumed: the repo includes an evaluation harness, a hand-checked question set, and scripts that compare retrieval settings.

## Key features

- **Hybrid retrieval (FAISS + BM25)**: dense semantic search plus keyword search combined with LangChain's `EnsembleRetriever`.
- **Source pages under answers**: each document-based answer shows `Sources retrieved: <file> (PDF pages ...)`, built from the pages of the chunks the retriever returned for that turn.
- **Human-in-the-loop approval for web search**: every web search is paused with LangGraph's `interrupt()` and needs an explicit Approve or Reject.
- **Tool failures do not crash the app**: if a tool raises (for example web search with no network), the error is returned to the model as a message instead of ending the run.
- **Model errors become plain messages**: a quota, rate-limit or outage error from the LLM shows a short readable message in the chat instead of a traceback, and the conversation stays usable.
- **Streamed responses** for normal turns, filtered to the correct graph node.
- **Conversation persistence** with `SqliteSaver`, including turns that are paused for approval.
- **Async PDF ingestion** on a background thread with live progress in the sidebar.
- **In-memory query caching** of retrieval results.
- **Swappable LLM backend**: the model is chosen in one function and can be overridden with an environment variable.
- **Input guardrail**: a lightweight LLM classifier screens clearly abusive, illegal or spam input before the main flow.
- **Automated tests** (pytest) covering the calculator, guardrail parsing, caching, source extraction, tool-failure handling, and the setup patch, all independent of live API calls.
- **Evaluation harness** (RAGAS plus retrieval hit-rate scripts); see [Evaluation](#evaluation).

## Architecture

```
START
  |
  v
guardrail_node --(inappropriate query)--> END (refusal message)
  |
  v (query passes)
chat_node <-------------------------------+
  |                                        |
  |-(no tool needed)--> END                |
  |                                        |
  |-(wants web search)--> human_review_node|
  |                         |-approved--> tools --+
  |                         '-rejected--> back to chat_node with a denial message
  |                                        |
  '-(calculator / document search)--> tools
```

The graph is built with LangGraph. `chat_node` is the only node that writes the real answer; `guardrail_node` runs a separate, cheap classification call first. Only `duckduckgo_search` goes through `human_review_node`; the calculator and the document-retrieval tool are deterministic and internal, so they run without approval.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) | Explicit state machine with native human-in-the-loop interrupts |
| LLM | Google Gemini (free tier) | No per-call cost during development |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Free, offline, no API dependency for retrieval |
| Vector store | FAISS | Fast local similarity search |
| Lexical search | BM25 (`rank_bm25`) | Catches exact keyword matches dense embeddings can miss |
| Persistence | SQLite via `SqliteSaver` | Zero-setup conversation storage |
| Web search | DuckDuckGo (`ddgs`) | No API key required |
| Evaluation | [RAGAS](https://github.com/explodinggradients/ragas) | LLM-judged RAG metrics |
| Observability | LangSmith | Trace-level view of every graph run |
| Frontend | Streamlit | Fast to build |
| Testing | pytest | Standard |

## Evaluation

All numbers below come from runs of the scripts in this repo. They are small-sample results and are labelled with their limits.

### Question set

24 hand-checked questions over three different documents: a machine-learning textbook (10 questions), a research paper (8) and a practical project guide (6). Every question has a ground-truth answer and an `evidence` phrase copied from the PDF; `verify_questions.py` checks that each phrase really appears in the extracted text. Each question runs on its own fresh conversation thread, and the retrieved context that is scored is the chunks the chatbot's own `rag_tool` calls returned, not a separate re-retrieval.

### End-to-end quality (RAGAS, hybrid retrieval, single run)

| Document | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|
| Textbook (10 q) | 1.0 | 0.80 | 0.72 |
| Research paper (8 q) | 1.0 | 0.90 | 0.84 |
| Project guide (6 q) | 1.0 | 0.94 | 0.70 |
| **All 24** | **1.0** | **0.87** | **0.75** |

How to read this:
- **Faithfulness is saturated at 1.0.** Answers stayed inside the retrieved text, which is what the system prompt asks for, but a metric at its ceiling cannot separate configurations, so it is not used to compare them.
- The questions are single-fact lookups, and 7 of the 10 textbook questions were written from passages that earlier runs had already retrieved, so the set is on the easy side.
- One run, judged by the same model family that writes the answers; treat differences of a few hundredths as noise.

### Retrieval comparison (no LLM calls, deterministic)

Each raw question is used as the query, and a question counts as a hit if a retrieved chunk contains its evidence phrase.

| Config | Hit rate | MRR |
|---|---|---|
| FAISS only, k=6 | 18/24 | 0.503 |
| FAISS only, k=4 | 18/24 | 0.503 |
| **Hybrid (k=4 + BM25 k=2), current** | **20/24** | 0.510 |

Hybrid found the evidence for two more questions; ranking quality was about equal. This measures retrieval only: it does not measure how much irrelevant text comes along with the right chunk. In the app the model rewrites queries and may search more than once, so the in-app hit rate was higher than this stricter test (23/24 in the full run).

### Chunk size and a noise filter (measured, not adopted)

Scored by whether any retrieved chunk comes from a page that contains the evidence, which is fair across chunk sizes.

| Config | Page hit rate | Noise share of retrieved chunks |
|---|---|---|
| **1000 / 200 (current)** | **22/24** | 0.7% |
| 1000 / 200 + noise filter | 22/24 | 0.0% |
| 500 / 100 | 20/24 | 0.0% |
| 1500 / 300 | 20/24 | 1.5% |
| 1000 / 0 (no overlap) | 21/24 | 2.4% |

The current 1000/200 setting scored best, so it stays. A filter that drops table-of-contents, index and divider chunks (about 5% of the textbook) cut retrieved noise from 0.7% to 0% with the same hit rate; it was not adopted because the gain is marginal and it adds risk on layouts it has not been tested on. The code is in `chunk_filter.py` with tests.

### Guardrail accuracy

On a hand-written set of 40 queries (20 that should be allowed, 20 that should be blocked): all 20 allowed queries were allowed, and 18 of 20 blocked queries were blocked (all abusive, illegal-request and promotional-spam cases). The two misses were gibberish strings (`asdf asdf ...`, a run of `a`s), which the deliberately permissive prompt does not treat as spam. The cases are mine, so they are on the easy side.

### Cache speed (retrieval step only)

A cache hit takes about 0.07 ms against a median of about 14 ms uncached on the 2,339-chunk textbook (about 190x over 10 queries). Across three test documents the ratio ranged from 65x to 432x, so treat it as an order-of-magnitude figure. This is the retrieval step only: the Gemini call (typically 1 to 4 s) dominates real response time, and the cache does not store LLM output.

## Key design decisions and trade-offs

**Hybrid retrieval tuning: reweighting alone does not work.** `EnsembleRetriever` merges the union of each retriever's results and only re-ranks by weight. Changing `weights=[0.5, 0.5]` to `[0.8, 0.2]` left the candidate set unchanged. The BM25 half was surfacing an "Exercises" section as keyword false positives, and the fix was reducing BM25's own `k` (`bm25_k=2`), not the weights.

**Human-in-the-loop gates only web search.** Only `duckduckgo_search` needs approval. The calculator and document retrieval are deterministic and internal; gating them would add friction with no safety benefit.

**Guardrails are prompt-based.** Input filtering is one cheap YES/NO LLM call, and groundedness is enforced by the system prompt rather than a separate verifier. Simpler to build and explain, at the cost of being less robust than a purpose-trained model.

**Caching covers retrieval, not LLM output.** Cached answers could go stale when the system prompt changes, so only the deterministic retrieval step is cached.

**One document per conversation thread, with a replace confirmation.** Uploading a second PDF to a thread that already has one asks for explicit confirmation instead of silently overwriting. Multiple documents per thread were deliberately deferred.

**Streaming covers new turns; the approval-resume path does not.** Resumed responses are short, and keeping the two code paths separate keeps each one simple.

**Local embeddings.** `all-MiniLM-L6-v2` runs locally with no per-call cost and removes one network dependency from the retrieval path.

## Project structure

```
RAG_Chat_Bot/
├── langgraph_backend.py       # Graph: nodes, routing, tools, guardrail, tool-failure handling
├── streamlit_frontend.py      # Streamlit UI: chat, PDF upload, approval flow, streaming, sources
├── rag_utils.py               # PDF ingestion, hybrid retriever, rag_tool, caching
├── source_utils.py            # Extracts and formats the pages a turn's answer was retrieved from
├── error_utils.py             # Turns model errors into short plain-language messages
├── llm_config.py              # Single place that selects the LLM
├── patch_ragas.py             # One-command fix for the ragas import problem (see Setup)
├── evaluation.py              # RAGAS evaluation over eval_questions/
├── eval_questions/            # Hand-checked questions with evidence phrases (one file per document)
├── verify_questions.py        # Confirms every evidence phrase exists in its PDF
├── compare_retrieval.py       # FAISS-only vs hybrid retrieval comparison
├── hit_rate.py                # Evidence hit rate for a saved evaluation run
├── compare_chunking.py        # Chunk size and noise-filter comparison
├── chunk_filter.py            # Contents/index/divider chunk detector (measured, not adopted)
├── bench_cache.py             # Cache timing benchmark
├── guardrail_eval.py          # Guardrail accuracy test
├── guardrail_cases.json       # The 40 guardrail test queries
├── test_*.py                  # pytest suites
├── requirements.txt
└── .gitignore
```

## Setup

**Requirements:** Python 3.11+ (developed on 3.14) and a free [Google AI Studio](https://aistudio.google.com/) API key.

```bash
git clone https://github.com/d4t4forge-debugX/RAG_Chat_Bot.git
cd RAG_Chat_Bot

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python patch_ragas.py
```

**Known setup issue:** `ragas==0.3.9` imports a module that no longer exists in current `langchain-community`, so `import ragas` fails on a fresh install. `python patch_ragas.py` applies a small, idempotent fix to the installed package, keeps a backup, and checks that `import ragas` works afterwards. Only the evaluation scripts need ragas; the app itself does not.

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Optional, to use a different Gemini model without editing code:

```
LLM_MODEL=your_model_name
LLM_THINKING_LEVEL=low
```

## Running it

```bash
streamlit run streamlit_frontend.py
```

Open the local URL Streamlit prints, upload a PDF from the sidebar, and ask questions.

## Running the tests

```bash
pytest -v
```

44 tests currently pass, covering the calculator, guardrail parsing, caching, source extraction, tool-failure handling, the setup patch, the chunk filter, error messages, and the UI's handling of model failures. They use mocks and fake messages, so they make no LLM calls and need no PDFs.

## Running the evaluation

The evaluation PDFs are not included in the repository (the textbook is copyrighted, and the other two are third-party documents). `eval_questions/*.json` refer to them by file name; to reproduce the numbers, place equivalent PDFs in the project root under those names, then:

```bash
python verify_questions.py            # every evidence phrase exists in its PDF
python evaluation.py --label hybrid   # RAGAS run, saved under eval_runs/
python hit_rate.py                    # evidence hit rate of the latest run
python compare_retrieval.py           # FAISS-only vs hybrid
python compare_chunking.py            # chunk sizes and the noise filter
python bench_cache.py                 # cache timing
python guardrail_eval.py              # guardrail accuracy
```

`evaluation.py`, `guardrail_eval.py` and the other scripts that call the model make many requests. On the Gemini free tier (500 requests per day and 15 per minute for the default model at the time of writing) a full evaluation can use most of a day's quota.

## Known limitations

Documented rather than left implicit:

- **No authentication or multi-user support**; this is a single-user local project.
- **In-memory per-thread state resets on restart**: retrievers and query caches are not persisted. A production version would use a persistent vector store.
- **One document per conversation thread.**
- **Sources are shown for live answers only**: they are not restored when a saved thread is reloaded, and a follow-up answered from earlier conversation text (with no new retrieval) shows no sources. The label says "retrieved", not "used": some retrieved pages may not have contributed to the answer.
- **LLM errors are reported, not retried**: a quota, rate-limit or outage error shows a short plain-language message in the chat (tested with simulated errors, not yet checked against a live outage) and the conversation stays usable. Nothing retries automatically, and the free-tier daily quota (500 requests for the default model at the time of writing) can run out during heavy evaluation.
- **No timeout on LLM calls**: one call hung once during testing.
- **Web search is best-effort**: the DuckDuckGo wrapper returned an off-topic result once and failed on some networks.
- **The guardrail is deliberately permissive**: it blocks clearly abusive, illegal or spam input and does not block gibberish.
- **No runtime hallucination gate**: groundedness relies on the system prompt plus offline evaluation, not a live verifier.
- **Evaluation limits**: 24 questions, mostly single-fact lookups, one run per measurement, an LLM judge from the same model family as the answerer, and a question set written by the author.
- **Not implemented, by choice**: semantic chunking, cost-aware model routing, multiple documents per thread.

## Roadmap

- [x] Hybrid retrieval, human-in-the-loop approval, persistence, streaming, async ingestion, caching
- [x] Larger evaluation set, retrieval comparisons, source pages in answers
- [x] README and project notes
- [ ] Harder evaluation questions (unanswerable and multi-passage)
- [x] Friendly handling of LLM errors in the UI
- [ ] Deployment (optional)

## License

MIT, see [LICENSE](LICENSE).