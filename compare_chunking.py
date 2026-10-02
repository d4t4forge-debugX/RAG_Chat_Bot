import glob
import json

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from chunk_filter import is_noise_chunk
from rag_utils import clean_text, embeddings, get_retriever


# lowercases and collapses all whitespace so line breaks don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# loads one PDF's pages once, so every config splits the same source text
def load_pages(path):
    return PyPDFLoader(path).load()


# splits pages the same way the app does (recursive splitter, cleaned text, empty chunks dropped), with a chosen size and overlap
def split_pages(pages, chunk_size, overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.page_content = clean_text(c.page_content)
    return [c for c in chunks if c.page_content.strip()]


# reads the verified questions with their evidence phrases
def load_questions():
    questions = []
    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            questions.extend(json.load(f))
    return questions


# the set of 0-based page numbers whose text contains the evidence phrase
def evidence_pages(pages, phrase):
    target = normalize(phrase)
    return {p.metadata.get("page") for p in pages if target in normalize(p.page_content)}


# runs one config over every question of one document; returns per-question (hit, noise share)
def run_config(pages, questions, chunk_size, overlap, use_filter):
    chunks = split_pages(pages, chunk_size, overlap)
    if use_filter:
        chunks = [c for c in chunks if not is_noise_chunk(c.page_content)]
    retriever = get_retriever(FAISS.from_documents(chunks, embeddings), chunks)

    rows = []
    for q in questions:
        wanted_pages = evidence_pages(pages, q["evidence"])
        returned = retriever.invoke(q["question"])
        hit = any(d.metadata.get("page") in wanted_pages for d in returned)
        noise = sum(1 for d in returned if is_noise_chunk(d.page_content)) / max(1, len(returned))
        rows.append((hit, noise, len(returned)))
    return rows, len(chunks)


def main():
    configs = {
        "current (1000/200, no filter)": (1000, 200, False),
        "filter (1000/200 + noise filter)": (1000, 200, True),
        "size500 (500/100)": (500, 100, False),
        "size1500 (1500/300)": (1500, 300, False),
        "no overlap (1000/0)": (1000, 0, False),
    }
    questions = load_questions()
    docs = sorted({q["doc"] for q in questions})

    totals = {name: [] for name in configs}
    for doc in docs:
        print(f"\nLoading {doc}...")
        pages = load_pages(doc)
        doc_questions = [q for q in questions if q["doc"] == doc]
        for name, (size, overlap, use_filter) in configs.items():
            rows, n_chunks = run_config(pages, doc_questions, size, overlap, use_filter)
            totals[name].extend(rows)
            hits = sum(1 for r in rows if r[0])
            print(f"   {name:36} {n_chunks:5} chunks | page hit {hits}/{len(rows)}")

    print("\n=== Overall (24 questions) ===")
    print(f"{'config':38} {'page hit rate':>14} {'noise share of retrieved':>26} {'avg chunks':>11}")
    for name, rows in totals.items():
        hits = sum(1 for r in rows if r[0])
        noise = sum(r[1] for r in rows) / len(rows)
        avg_chunks = sum(r[2] for r in rows) / len(rows)
        print(f"{name:38} {hits:>8}/{len(rows):<5} {noise * 100:>24.1f}% {avg_chunks:>11.1f}")


if __name__ == "__main__":
    main()