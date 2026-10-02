import statistics
import sys
import time
import glob
import json
import uuid

from rag_utils import ingest_pdf_for_thread, rag_tool, _QUERY_CACHE


# reads the verified eval questions for one document, to use as realistic queries
def questions_for(doc):
    questions = []
    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            questions += [q["question"] for q in json.load(f) if q["doc"] == doc]
    return questions


# times one rag_tool call in milliseconds
def timed_call(query, thread_id):
    start = time.perf_counter()
    result = rag_tool.invoke({"query": query, "thread_id": thread_id})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, result["cache_hit"]


# measures uncached vs cached retrieval time for one PDF and prints medians and the ratio
def bench(doc):
    thread_id = str(uuid.uuid4())
    print(f"\n=== {doc} ===")
    summary = ingest_pdf_for_thread(doc, thread_id=thread_id, filename=doc)
    print(f"{summary['chunks']} chunks indexed")

    queries = questions_for(doc)

    # warm-up: the first call loads the embedding model for queries, which would distort the uncached timing
    timed_call("warm up query that is not in the question set", thread_id)
    _QUERY_CACHE.clear()

    uncached = []
    for q in queries:
        ms, hit = timed_call(q, thread_id)
        assert hit is False, "expected a cache miss"
        uncached.append(ms)

    cached = []
    for i in range(20):
        ms, hit = timed_call(queries[i % len(queries)], thread_id)
        assert hit is True, "expected a cache hit"
        cached.append(ms)

    u, c = statistics.median(uncached), statistics.median(cached)
    print(f"uncached retrieval: median {u:.2f} ms over {len(uncached)} distinct queries")
    print(f"cached retrieval:   median {c:.4f} ms over {len(cached)} repeated calls")
    print(f"speedup (retrieval step only): {u / c:.0f}x")


if __name__ == "__main__":
    docs = sys.argv[1:] or ["test3.pdf", "test4.pdf"]
    for d in docs:
        bench(d)