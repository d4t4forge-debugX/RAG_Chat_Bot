import pytest

from rag_utils import demo_index_available, load_demo_for_thread, get_retriever_for_thread, _THREAD_METADATA, DEMO_DISPLAY_NAME

# these tests need the saved index; run `python build_demo_index.py` once to create it
needs_index = pytest.mark.skipif(not demo_index_available(), reason="demo index not built yet")


# loading the demo attaches a retriever and the display name to the thread, with no embedding step
@needs_index
def test_demo_loads_into_a_thread():
    result = load_demo_for_thread("demo-test-thread")
    assert get_retriever_for_thread("demo-test-thread") is not None
    assert _THREAD_METADATA["demo-test-thread"]["filename"] == DEMO_DISPLAY_NAME
    assert result["chunks"] > 0


# both regulations are in the saved index, and every chunk knows which file it came from
@needs_index
def test_demo_chunks_keep_their_source_file():
    load_demo_for_thread("demo-test-thread-2")
    retriever = get_retriever_for_thread("demo-test-thread-2")
    docs = retriever.invoke("lockout tagout device removal") + retriever.invoke("fall protection height above ground")
    names = {d.metadata.get("source_name") for d in docs}
    assert names <= {"osha_1910_147.pdf", "osha_1910_269.pdf"}
    assert "osha_1910_147.pdf" in names and "osha_1910_269.pdf" in names
