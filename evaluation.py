import argparse
import glob
import json
import math
import os
import time
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

from langgraph_backend import chatbot, llm
from rag_utils import ingest_pdf_for_thread, _THREAD_RETRIEVERS, _THREAD_METADATA
from rag_utils import embeddings as rag_embeddings

# the three RAGAS metric names as they appear in the scores object
METRIC_NAMES = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]


# reads every eval_questions/*.json file into one list, optionally keeping only the first N questions per file
def load_questions(limit_per_doc=None):
    questions = []
    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            items = json.load(f)
        if limit_per_doc:
            items = items[:limit_per_doc]
        questions.extend(items)
    return questions


# ingests each distinct PDF once and returns {pdf filename: base thread_id holding its retriever}
def ingest_documents(questions):
    base_threads = {}
    chunk_counts = {}
    for doc in sorted({q["doc"] for q in questions}):
        thread_id = str(uuid.uuid4())
        print(f"Ingesting {doc}...")
        summary = ingest_pdf_for_thread(doc, thread_id=thread_id, filename=doc)
        print(f"  {summary['chunks']} chunks")
        base_threads[doc] = thread_id
        chunk_counts[doc] = summary["chunks"]
    return base_threads, chunk_counts


# makes a brand-new thread that reuses an already-built retriever, so each question gets a clean chat history without re-embedding the PDF
def create_fresh_thread_with_document(source_thread_id: str) -> str:
    new_thread_id = str(uuid.uuid4())
    _THREAD_RETRIEVERS[new_thread_id] = _THREAD_RETRIEVERS[str(source_thread_id)]
    _THREAD_METADATA[new_thread_id] = _THREAD_METADATA[str(source_thread_id)]
    return new_thread_id


# pulls out the chunks the chatbot's own rag_tool calls returned for the latest question, instead of re-running the retriever
def get_contexts_used_by_chatbot(messages):
    contexts = []
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and msg.name == "rag_tool":
            try:
                data = json.loads(msg.content)
            except (TypeError, json.JSONDecodeError):
                continue
            for chunk in data.get("context", []):
                if chunk not in contexts:
                    contexts.append(chunk)
    return contexts


# turns a message's content (plain string or list of content blocks) into plain text
def content_to_text(content):
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return content or ""


# asks the chatbot one question on a fresh thread; retries on errors (e.g. rate limits), returning None if every attempt fails
def ask_chatbot(question, base_thread_id, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            thread_id = create_fresh_thread_with_document(base_thread_id)
            return chatbot.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"configurable": {"thread_id": thread_id}},
            )
        except Exception as e:
            print(f"  attempt {attempt}/{attempts} failed: {e}")
            time.sleep(20 * attempt)
    return None


# runs every question through the real chatbot and records answer, retrieved chunks, and a status explaining any problem
def run_evaluation(questions, base_threads):
    records = []
    for i, item in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] ({item['doc']}) {item['question']}")
        record = {
            "doc": item["doc"],
            "question": item["question"],
            "ground_truth": item["ground_truth"],
            "answer": "",
            "contexts": [],
            "status": "ok",
        }

        response = ask_chatbot(item["question"], base_threads[item["doc"]])
        if response is None:
            record["status"] = "chatbot_error"
        elif "__interrupt__" in response:
            record["status"] = "paused_for_approval"
        else:
            record["answer"] = content_to_text(response["messages"][-1].content).strip()
            record["contexts"] = get_contexts_used_by_chatbot(response["messages"])
            if not record["answer"]:
                record["status"] = "empty_answer"
            elif not record["contexts"]:
                record["status"] = "no_retrieval"

        if record["status"] != "ok":
            print(f"  WARNING: status = {record['status']} (not scored)")
        records.append(record)
    return records


# averages a list of scores, ignoring missing/NaN values (RAGAS returns NaN when a judge call fails)
def mean_ignoring_nan(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return sum(clean) / len(clean) if clean else None


# feeds the usable records into RAGAS and attaches each question's own three scores back onto its record
def score_with_ragas(records):
    ok_records = [r for r in records if r["status"] == "ok"]
    if not ok_records:
        return {}

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in ok_records
    ])

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(strictness=1), LLMContextPrecisionWithReference()],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(rag_embeddings),
        run_config=RunConfig(timeout=180, max_workers=1),
    )

    for idx, record in enumerate(ok_records):
        record["scores"] = {name: scores[name][idx] for name in METRIC_NAMES}

    return {name: mean_ignoring_nan(scores[name]) for name in METRIC_NAMES}


# entry point: loads questions, ingests PDFs, runs and scores them, and saves everything to eval_runs/<label>_<timestamp>.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-doc", type=int, default=None,
                        help="only use the first N questions from each question file (cheap smoke test)")
    parser.add_argument("--label", default="hybrid_k4_bm25k2",
                        help="name for this run's configuration, used in the output filename")
    args = parser.parse_args()

    questions = load_questions(args.limit_per_doc)
    print(f"Loaded {len(questions)} questions.\n")

    base_threads, chunk_counts = ingest_documents(questions)
    print()

    records = run_evaluation(questions, base_threads)

    print("\nRunning RAGAS evaluation...")
    mean_scores = score_with_ragas(records)

    status_counts = {}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    print("\n=== Mean scores (ignoring failed judge calls) ===")
    for name, value in mean_scores.items():
        print(f"{name}: {value}")
    print("\nQuestion statuses:", status_counts)

    os.makedirs("eval_runs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"eval_runs/{args.label}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "label": args.label,
                "timestamp": timestamp,
                "model": getattr(llm, "model", "unknown"),
                "questions_total": len(questions),
                "documents": chunk_counts,
                "status_counts": status_counts,
            },
            "mean_scores": mean_scores,
            "questions": records,
        }, f, indent=2)

    print(f"\nSaved results to {out_path}")