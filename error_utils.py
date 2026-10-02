import re

# Gemini's quota errors include text like "Please retry in 8h57m50.969470679s"
RETRY_IN = re.compile(r"retry in ([0-9hms\.]+)", re.IGNORECASE)


# shortens a duration like "8h57m50.969s" to "8h 57m", or "3.558s" to "4s"
def _tidy_wait(raw):
    hours = re.search(r"(\d+)h", raw)
    minutes = re.search(r"(\d+)m(?!s)", raw)
    seconds = re.search(r"([\d\.]+)s", raw)
    if hours or minutes:
        parts = []
        if hours:
            parts.append(f"{hours.group(1)}h")
        if minutes:
            parts.append(f"{minutes.group(1)}m")
        return " ".join(parts)
    if seconds:
        return f"{round(float(seconds.group(1)))}s"
    return raw


# turns an exception from a model call into one short, readable sentence for the chat window (never the raw API dump)
def friendly_error_message(error):
    text = f"{type(error).__name__}: {error}"
    lowered = text.lower()

    if "resource_exhausted" in lowered or re.search(r"\b429\b", lowered) or "ratelimit" in lowered:
        retry = RETRY_IN.search(text)
        wait = f" The service says to retry in about {_tidy_wait(retry.group(1))}." if retry else ""
        if "perday" in lowered:
            return "The model's daily free-tier quota is used up." + wait + " Try again later."
        return "The model is rate-limited right now." + wait + " Wait a moment and try again."

    if "unavailable" in lowered or re.search(r"\b503\b", lowered):
        return "The model service is temporarily unavailable. Please try again in a minute."

    if any(word in lowered for word in ("dns", "connect", "timed out", "timeout", "network")):
        return "Could not reach the model service. Check your internet connection and try again."

    return "Something went wrong while generating the answer. Please try again."