import os
import re
import threading
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Stores each thread's vector store, keyed by thread_id
_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}

# Simple in-memory cache for rag_tool results.
# Key: (thread_id, query) -> avoids re-embedding + re-searching for a
# question that's already been asked in this session. Cleared on app restart
# (in-memory only), same limitation as the retriever store above.
_QUERY_CACHE = {}

# Tracks background ingestion progress, keyed by thread_id.
# status: "running" | "done" | "error"
_INGESTION_STATUS = {}


# wipes any cached rag_tool answers for one thread so a re-ingested PDF can't be shadowed by stale cached results
def _clear_thread_cache(thread_id: str):
    """
    Drop any cached rag_tool results for this thread. Called whenever a
    (re-)ingestion completes, so a newly uploaded PDF can't be shadowed by
    stale cached answers from a previous document in the same thread.
    """
    key = str(thread_id)
    stale_keys = [k for k in _QUERY_CACHE if k[0] == key]
    for k in stale_keys:
        del _QUERY_CACHE[k]


# blocking end-to-end PDF ingestion: load, split, embed, build hybrid retriever, and store it for the thread
def ingest_pdf_for_thread(pdf_path: str, thread_id: str, filename: str = None):
    """Synchronous ingestion (kept for evaluation.py and the __main__ test below)."""
    chunks = load_and_split_pdf(pdf_path)
    vector_store = build_vector_store(chunks)
    retriever = build_thread_retriever(vector_store, chunks)

    _THREAD_RETRIEVERS[str(thread_id)] = retriever
    _THREAD_METADATA[str(thread_id)] = {
        "filename": filename or pdf_path,
        "chunks": len(chunks),
    }
    _clear_thread_cache(thread_id)

    return {"filename": filename or pdf_path, "chunks": len(chunks)}


# same ingestion pipeline as above, but runs on a background thread and reports live stage updates via _INGESTION_STATUS
def _ingest_pdf_worker(pdf_path: str, thread_id: str, filename: str):
    """Runs on a background thread. Updates _INGESTION_STATUS as it progresses."""
    key = str(thread_id)
    try:
        _INGESTION_STATUS[key] = {"status": "running", "stage": "Loading PDF..."}
        chunks = load_and_split_pdf(pdf_path)

        _INGESTION_STATUS[key] = {"status": "running", "stage": f"Building embeddings for {len(chunks)} chunks..."}
        vector_store = build_vector_store(chunks)

        _INGESTION_STATUS[key] = {"status": "running", "stage": "Building hybrid retriever..."}
        retriever = build_thread_retriever(vector_store, chunks)

        _THREAD_RETRIEVERS[key] = retriever
        _THREAD_METADATA[key] = {"filename": filename or pdf_path, "chunks": len(chunks)}
        _clear_thread_cache(key)

        _INGESTION_STATUS[key] = {
            "status": "done",
            "stage": "Done",
            "result": {"filename": filename or pdf_path, "chunks": len(chunks)},
        }
    except Exception as e:
        _INGESTION_STATUS[key] = {"status": "error", "stage": "Error", "error": str(e)}


# fire-and-forget launcher: spins up the background ingestion thread so the caller (Streamlit) doesn't block
def start_ingestion_async(pdf_path: str, thread_id: str, filename: str = None):
    """
    Kicks off PDF ingestion on a background thread so the caller isn't
    blocked. Progress can be checked via get_ingestion_status(thread_id).
    """
    thread = threading.Thread(
        target=_ingest_pdf_worker,
        args=(pdf_path, thread_id, filename),
        daemon=True,
    )
    thread.start()


# polling helper: lets the UI check current ingestion stage/status for a thread without blocking
def get_ingestion_status(thread_id: str):
    return _INGESTION_STATUS.get(str(thread_id))


# lookup helper: returns the stored hybrid retriever for a thread, or None if nothing's been ingested yet
def get_retriever_for_thread(thread_id: str):
    return _THREAD_RETRIEVERS.get(str(thread_id))


# strips orphaned unicode surrogate characters that can crash the embedding tokenizer on math-heavy PDF text
def clean_text(text: str) -> str:
    # Remove invalid/orphaned unicode surrogate characters
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text


# loads a PDF, splits it into overlapping chunks, cleans each chunk's text, and drops any empty chunks
def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Clean each chunk's text and filter out empty ones
    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

    return chunks


# shared local HuggingFace embedding model instance used for all FAISS vector store builds
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# builds a FAISS vector store (semantic/dense retrieval) from the given document chunks
def build_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# builds a BM25 retriever (lexical/keyword retrieval) from the given document chunks, capped to top-k results
def build_bm25_retriever(chunks, k=4):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return bm25_retriever


# combines FAISS + BM25 into one weighted hybrid retriever via reciprocal rank fusion (0.7 semantic / 0.3 lexical)
def get_retriever(vector_store, chunks, k=4, bm25_k=2):
    faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    bm25_retriever = build_bm25_retriever(chunks, k=bm25_k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return hybrid_retriever


# switch for A/B runs: True = wide hybrid pool reranked by the cross-encoder, False = the original hybrid retriever
USE_RERANKER = True
# how many chunks survive reranking (top-4 beat top-3 and the current hybrid in compare_rerank.py)
RERANK_TOP_N = 4

# small local cross-encoder that scores each (question, chunk) pair together; only loaded once
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


# "retrieve wide, rerank narrow": up to 15 hybrid candidates (FAISS k=10 + BM25 k=5), cross-encoder keeps the best RERANK_TOP_N
def get_reranked_retriever(vector_store, chunks, top_n=RERANK_TOP_N):
    wide_pool = get_retriever(vector_store, chunks, k=10, bm25_k=5)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=wide_pool)


# the retriever the app actually uses for a thread, chosen by the USE_RERANKER switch
def build_thread_retriever(vector_store, chunks):
    if USE_RERANKER:
        return get_reranked_retriever(vector_store, chunks)
    return get_retriever(vector_store, chunks)


# ---- Sample energy-safety corpus (public-domain OSHA regulations) ----
# the two PDFs that make up the demo, and where their pre-built FAISS index is saved (built once by build_demo_index.py)
DEMO_PDFS = ["demo_corpus/energy/osha_1910_147.pdf", "demo_corpus/energy/osha_1910_269.pdf"]
DEMO_INDEX_DIR = "demo_corpus/energy/index"
DEMO_DISPLAY_NAME = "OSHA energy safety rules (1910.147 + 1910.269)"


# loads both demo PDFs into one chunk list, tagging every chunk with the file it came from (so sources can name the right regulation)
def load_demo_chunks():
    chunks = []
    for pdf_path in DEMO_PDFS:
        for chunk in load_and_split_pdf(pdf_path):
            chunk.metadata["source_name"] = os.path.basename(pdf_path)
            chunks.append(chunk)
    return chunks


# True once build_demo_index.py has been run and the saved index exists on disk
def demo_index_available():
    return os.path.exists(os.path.join(DEMO_INDEX_DIR, "index.faiss"))


# attaches the pre-built demo index to a chat thread instantly: no embedding at click time, which matters on a slow free-tier server
def load_demo_for_thread(thread_id: str):
    key = str(thread_id)
    # allow_dangerous_deserialization: FAISS stores chunk text in a pickle file; safe here only because we built this file ourselves
    vector_store = FAISS.load_local(DEMO_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    # BM25 and the reranker need the chunk list itself, which the saved index already holds
    chunks = [vector_store.docstore.search(doc_id) for doc_id in vector_store.index_to_docstore_id.values()]

    _THREAD_RETRIEVERS[key] = build_thread_retriever(vector_store, chunks)
    _THREAD_METADATA[key] = {"filename": DEMO_DISPLAY_NAME, "chunks": len(chunks)}
    _clear_thread_cache(key)
    result = {"filename": DEMO_DISPLAY_NAME, "chunks": len(chunks)}
    # reuse the ingestion status so the sidebar shows the same "Indexed ..." message as a normal upload
    _INGESTION_STATUS[key] = {"status": "done", "stage": "Done", "result": result}
    return result


# LLM-callable tool: retrieves relevant chunks for a thread's PDF, serving from cache when the same query repeats
@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Use this whenever the user asks a question that could be answered by their uploaded document.
    """
    # thread_id comes from the run's own config (injected by LangGraph), never from the model,
    # so the model can't mistype it or point the tool at another conversation's document
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    cache_key = (str(thread_id), query.strip().lower())
    if cache_key in _QUERY_CACHE:
        cached_result = dict(_QUERY_CACHE[cache_key])
        cached_result["cache_hit"] = True
        return cached_result

    retriever = get_retriever_for_thread(thread_id)
    if retriever is None:
        return {
            "error": "No document has been uploaded for this chat yet. Ask the user to upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)
    context = [doc.page_content for doc in results]
    pages = [doc.metadata.get("page") for doc in results]
    thread_filename = _THREAD_METADATA.get(str(thread_id), {}).get("filename")
    # per-chunk file + page, so a multi-file corpus (the OSHA sample) can show which document each page came from
    sources = [
        {"file": doc.metadata.get("source_name") or thread_filename, "page": doc.metadata.get("page")}
        for doc in results
    ]

    result = {
        "query": query,
        "context": context,
        "pages": pages,
        "sources": sources,
        "source_file": thread_filename,
        "cache_hit": False,
    }

    _QUERY_CACHE[cache_key] = result
    return result


# standalone smoke test: ingests test.pdf directly and prints top hybrid-retrieval results for a sample query
if __name__ == "__main__":
    chunks = load_and_split_pdf("test.pdf")
    print(f"Total chunks created: {len(chunks)}")

    print("Building vector store with ALL chunks...")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully!")

    retriever = build_thread_retriever(vector_store, chunks)
    query = "What is supervised learning?"
    results = retriever.invoke(query)

    print(f"\nTop {len(results)} results for query: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (page {doc.metadata.get('page')}) ---")
        print(doc.page_content[:300])
        print()