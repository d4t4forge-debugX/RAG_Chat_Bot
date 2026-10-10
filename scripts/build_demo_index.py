from rag_utils import load_demo_chunks, build_vector_store, DEMO_INDEX_DIR

# Run once (and again only if the demo PDFs change):  python -m scripts.build_demo_index
# Embeds both OSHA PDFs locally and saves the FAISS index, so the app can load the demo instantly
# instead of embedding ~90 pages on a slow free-tier server.


# builds and saves the demo index, then prints a short summary per source file
def main():
    chunks = load_demo_chunks()
    per_file = {}
    for chunk in chunks:
        per_file[chunk.metadata["source_name"]] = per_file.get(chunk.metadata["source_name"], 0) + 1
    print(f"{len(chunks)} chunks: {per_file}")

    vector_store = build_vector_store(chunks)
    vector_store.save_local(DEMO_INDEX_DIR)
    print(f"Saved index to {DEMO_INDEX_DIR}/ (index.faiss + index.pkl)")


if __name__ == "__main__":
    main()
