import glob
import json
import os
from datetime import datetime

from rag_utils import load_and_split_pdf, build_vector_store, get_retriever


# lowercases and collapses all whitespace so line breaks in chunk text don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# reads every verified question (with its doc and evidence phrase) from eval_questions/*.json
def load_questions():
    questions = []
    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            questions.extend(json.load(f))
    return questions


# builds the three retrieval configs being compared over one PDF's chunks (same embeddings, same vector store)
def build_retrievers(pdf_path):
    chunks = load_and_split_pdf(pdf_path)
    vector_store = build_vector_store(chunks)
    return {
        "faiss_k6": vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6}),
        "faiss_k4": vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        "hybrid_k4_bm25k2": get_retriever(vector_store, chunks, k=4, bm25_k=2),
    }, len(chunks)


# returns the 1-based position of the first returned chunk containing the evidence phrase, or None
def first_hit_position(texts, phrase):
    target = normalize(phrase)
    for position, text in enumerate(texts, start=1):
        if target in normalize(text):
            return position
    return None


# runs every question through every config and prints hit rate, MRR and average chunk count per doc and overall
def main():
    questions = load_questions()
    docs = sorted({q["doc"] for q in questions})
    config_names = ["faiss_k6", "faiss_k4", "hybrid_k4_bm25k2"]

    # results[config] = list of (doc, question, position or None, chunks returned)
    results = {name: [] for name in config_names}

    for doc in docs:
        print(f"Building retrievers for {doc}...")
        retrievers, n_chunks = build_retrievers(doc)
        print(f"  {n_chunks} chunks")
        for q in [x for x in questions if x["doc"] == doc]:
            for name in config_names:
                returned = retrievers[name].invoke(q["question"])
                texts = [d.page_content for d in returned]
                position = first_hit_position(texts, q["evidence"])
                results[name].append((doc, q["question"], position, len(texts)))

    print()
    for name in config_names:
        rows = results[name]
        print(f"=== {name} ===")
        for doc in docs + ["ALL"]:
            subset = rows if doc == "ALL" else [r for r in rows if r[0] == doc]
            hits = [r for r in subset if r[2] is not None]
            mrr = sum(1 / r[2] for r in hits) / len(subset)
            avg_chunks = sum(r[3] for r in subset) / len(subset)
            print(f"  {doc:10} hit rate {len(hits)}/{len(subset)} | MRR {mrr:.3f} | avg chunks returned {avg_chunks:.1f}")
        misses = [r for r in rows if r[2] is None]
        for doc, question, _, n in misses:
            print(f"     MISS: {doc} | {question[:70]}")
        print()

    os.makedirs("eval_runs", exist_ok=True)
    out_path = f"eval_runs/retrieval_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(
            {name: [{"doc": d, "question": q, "first_hit_position": p, "chunks": n} for d, q, p, n in rows]
             for name, rows in results.items()},
            f, indent=2,
        )
    print("Saved to", out_path)


if __name__ == "__main__":
    main()