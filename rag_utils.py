from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
import re

# Stores each thread's vector store, keyed by thread_id
_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}

def ingest_pdf_for_thread(pdf_path: str, thread_id: str, filename: str = None):
    chunks = load_and_split_pdf(pdf_path)
    vector_store = build_vector_store(chunks)
    retriever = get_retriever(vector_store, chunks)

    _THREAD_RETRIEVERS[str(thread_id)] = retriever
    _THREAD_METADATA[str(thread_id)] = {
        "filename": filename or pdf_path,
        "chunks": len(chunks),
    }

    return {"filename": filename or pdf_path, "chunks": len(chunks)}


def get_retriever_for_thread(thread_id: str):
    return _THREAD_RETRIEVERS.get(str(thread_id))

def clean_text(text: str) -> str:
    # Remove invalid/orphaned unicode surrogate characters
    text = re.sub(r'[\ud800-\udfff]', '', text)
    return text


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

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def build_vector_store(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def build_bm25_retriever(chunks, k=4):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    return bm25_retriever

def get_retriever(vector_store, chunks, k=4, bm25_k=2):
    faiss_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    bm25_retriever = build_bm25_retriever(chunks, k=bm25_k)

    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    return hybrid_retriever

if __name__ == "__main__":
    chunks = load_and_split_pdf("test.pdf")
    print(f"Total chunks created: {len(chunks)}")

    print("Building vector store with ALL chunks...")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully!")

    retriever = get_retriever(vector_store, chunks)
    query = "What is supervised learning?"
    results = retriever.invoke(query)

    print(f"\nTop {len(results)} results for query: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} (page {doc.metadata.get('page')}) ---")
        print(doc.page_content[:300])
        print()

from langchain_core.tools import tool
from typing import Optional


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    Use this whenever the user asks a question that could be answered by their uploaded document.
    """
    retriever = get_retriever_for_thread(thread_id)
    if retriever is None:
        return {
            "error": "No document has been uploaded for this chat yet. Ask the user to upload a PDF first.",
            "query": query,
        }

    results = retriever.invoke(query)
    context = [doc.page_content for doc in results]
    pages = [doc.metadata.get("page") for doc in results]

    return {
        "query": query,
        "context": context,
        "pages": pages,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }