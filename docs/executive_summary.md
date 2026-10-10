# RAG Chatbot: one-page summary

## Problem
People who work from long technical documents (regulations, manuals, papers) need precise answers with a page they can check, not a fluent guess. A general chatbot answers from memory and cannot show where an answer came from.

## Users
Anyone who uploads a PDF and asks questions about it. The built-in sample targets field and safety staff in the energy sector: it loads two US workplace-safety regulations, OSHA 29 CFR 1910.147 (lockout/tagout) and 1910.269 (electric power generation, transmission and distribution).

## Outcome measured
- **Answers stay inside the documents**: faithfulness (RAGAS) and correct refusals on questions the documents cannot answer.
- **Retrieval brings back the right text, ranked first**: context precision (RAGAS) and mean reciprocal rank on evidence phrases.
- **Nothing leaves the system without a person's approval**: every web search needs an explicit click.

## How it works
1. A cheap classifier screens the question.
2. The model searches the document: about 15 candidates from dense and keyword search, reranked by a local cross-encoder to the best 4.
3. It answers only from what was retrieved and lists the source pages per file.
4. A web search, if the model wants one, pauses for human approval.
5. Conversations persist in SQLite; every run is traced in LangSmith.

## Results (small samples, single runs)
| Measure | Result |
|---|---|
| Context precision | 0.75 → 0.86 with the reranker, 24 questions, higher for all three documents |
| Faithfulness | 1.0 → 0.99 |
| Answer relevancy | 0.87 → 0.81 (mostly correct but terse answers scored lower) |
| Retrieval ranking (MRR, no LLM) | 0.490 → 0.645 with the reranker, 27 questions |
| Unanswerable questions refused | 4/4 |
| Prompt injection planted in a document | 0/4 attacks changed the answer |
| Energy-safety set (11 questions, OSHA rules) | faithfulness 1.0, context precision 0.76; evidence found 9/9; unanswerable refused 2/2 |
| Input guardrail (40 cases) | 39/40, no false blocks |
| Automated tests | 62, no model calls |

Three changes were measured and deliberately not shipped (an abstention step, an injection defence, a chunk-noise filter) because the data showed nothing for them to fix.

## Risks
- Evaluation sets are small and written by the author; the judge model is from the same family as the answering model.
- Answer relevancy dropped with the reranker, even though the lowest-scored answers read as correct.
- Web search results are not checked beyond the human approval step.
- Free-tier model quota limits daily use.

## Next steps
- Add a persistent vector store and a runtime grounding check.
