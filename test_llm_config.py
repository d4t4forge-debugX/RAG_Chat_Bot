from llm_config import get_llm, REQUEST_TIMEOUT_SECONDS, MAX_RETRIES


# the shared LLM must have a timeout (so a hung call can't freeze the app) and a small retry count (so 429s don't burn quota)
def test_llm_has_timeout_and_limited_retries(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
    llm = get_llm()
    assert llm.timeout == REQUEST_TIMEOUT_SECONDS == 60
    assert llm.max_retries == MAX_RETRIES == 2