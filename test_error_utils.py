from error_utils import friendly_error_message

DAILY_QUOTA = (
    "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 500, model: gemini-3.5-flash-lite "
    "Please retry in 8h57m50.969470679s. 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'"
)

PER_MINUTE_QUOTA = (
    "Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "limit: 15, model: gemini-3.5-flash-lite Please retry in 3.558505245s. "
    "'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'"
)


# verifies a daily-quota error says so and shows the shortened wait time
def test_daily_quota_message():
    message = friendly_error_message(RuntimeError(DAILY_QUOTA))
    assert "daily" in message
    assert "8h 57m" in message


# verifies a per-minute rate limit is described differently from a daily quota
def test_per_minute_rate_limit_message():
    message = friendly_error_message(RuntimeError(PER_MINUTE_QUOTA))
    assert "rate-limited" in message
    assert "4s" in message
    assert "daily" not in message


# verifies an overloaded-service error gets a retry-later message
def test_service_unavailable_message():
    message = friendly_error_message(RuntimeError("503 UNAVAILABLE. The service is currently unavailable."))
    assert "temporarily unavailable" in message


# verifies a DNS/connection failure tells the user to check their connection
def test_network_error_message():
    message = friendly_error_message(RuntimeError("error resolving DNS: no connections available"))
    assert "internet connection" in message


# verifies an unknown error falls back to a generic message
def test_unknown_error_gets_generic_message():
    assert "Something went wrong" in friendly_error_message(ValueError("odd failure"))


# verifies the raw API error code and details never reach the user
def test_raw_api_details_are_not_shown():
    message = friendly_error_message(RuntimeError(DAILY_QUOTA))
    assert "RESOURCE_EXHAUSTED" not in message
    assert "quotaId" not in message