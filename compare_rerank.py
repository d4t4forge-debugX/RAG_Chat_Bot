import glob
import json
import os
import time
from datetime import datetime

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from rag_utils import load_and_split_pdf, build_vector_store, get_retriever

# small local cross-encoder: scores each (question, chunk) pair together, so it can match different wordings of the same idea
CROSS_ENCODER = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


# lowercases and collapses whitespace so PDF line breaks don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# loads the 24 normal questions plus the answerable hard ones (unanswerables have no evidence to find, so they're skipped)
def load_questions():
    questions = []
    for path in sorted(glob.glob("eval_questions/*.json") + glob.glob("eval_questions_hard/*.json")):
        with open(path) as f:
            for item in json.load(f):
                evidence = item["evidence"] if isinstance(item["evidence"], list) else [item["evidence"]]
                if evidence:
                    questions.append({**item, "evidence": evidence})
    return questions


# wraps a wide hybrid pool (FAISS k=10 + BM25 k=5) with the cross-encoder, keeping only the top_n best chunks
def build_reranked(vector_store, chunks, top_n):
    wide_pool = get_retriever(vector_store, chunks, k=10, bm25_k=5)
    reranker = CrossEncoderReranker(model=CROSS_ENCODER, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=wide_pool)


# scores one retrieval: were ALL evidence phrases found, and at what position was the first one
def score(texts, evidence):
    normalized = [normalize(t) for t in texts]
    all_found = all(any(normalize(p) in t for t in normalized) for p in evidence)
    first = normalize(evidence[0])
    position = next((i for i, t in enumerate(normalized, start=1) if first in t), None)
    return all_found, position


# runs every question through each config and reports hit rate, MRR, chunks returned and time per query
def main():
    questions = load_questions()
    config_names = ["current_hybrid", "rerank_top4", "rerank_top3"]
    rows = {name: [] for name in config_names}

    for doc in sorted({q["doc"] for q in questions}):
        print(f"Building retrievers for {doc}...")
        chunks = load_and_split_pdf(doc)
        vector_store = build_vector_store(chunks)
        retrievers = {
            "current_hybrid": get_retriever(vector_store, chunks),
            "rerank_top4": build_reranked(vector_store, chunks, top_n=4),
            "rerank_top3": build_reranked(vector_store, chunks, top_n=3),
        }
        for q in [x for x in questions if x["doc"] == doc]:
            for name in config_names:
                start = time.perf_counter()
                returned = retrievers[name].invoke(q["question"])
                ms = (time.perf_counter() - start) * 1000
                all_found, position = score([d.page_content for d in returned], q["evidence"])
                rows[name].append({"doc": doc, "question": q["question"], "all_found": all_found,
                                   "position": position, "chunks": len(returned), "ms": round(ms, 1)})

    print()
    for name in config_names:
        r = rows[name]
        hits = sum(1 for x in r if x["all_found"])
        mrr = sum(1 / x["position"] for x in r if x["position"]) / len(r)
        avg_chunks = sum(x["chunks"] for x in r) / len(r)
        avg_ms = sum(x["ms"] for x in r) / len(r)
        print(f"{name:15} all evidence found {hits}/{len(r)} | MRR {mrr:.3f} | avg chunks {avg_chunks:.1f} | avg {avg_ms:.0f} ms")
        for x in r:
            if not x["all_found"]:
                print(f"    miss: ({x['doc']}) {x['question'][:70]}")

    os.makedirs("eval_runs", exist_ok=True)
    out_path = f"eval_runs/rerank_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()