import re

# chunks shorter than this (after trimming) are treated as page dividers, not content
MIN_CHARS = 80

# a line "ends like a page reference" if it finishes in a bare 1-4 digit number or a range like 192-196
PAGE_END = re.compile(r"\b\d{1,4}(?:-\d{1,4})?\s*$")

# decimal numbers (25.8, 4.66) mark results tables and equations; contents and index pages almost never have them
DECIMAL = re.compile(r"\b\d+\.\d+\b")


# returns the name of the rule that marks a chunk as noise (a divider, table of contents or index page), or None if it is content
def noise_reason(text):
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return "too short (divider)"

    # table-of-contents style dot leaders ("Preface. . . . . xv"): several runs of ". . . ." in one chunk
    if len(re.findall(r"(?:\.\s){4,}", stripped)) >= 3:
        return "dot leaders (table of contents)"

    # decimals mean a results table or equation block, which is content
    if len(DECIMAL.findall(stripped)) >= 5:
        return None

    # contents/index style: 6+ lines, most ending in a page reference, and the text is mostly letters
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 6:
        ends_with_page = sum(1 for line in lines if PAGE_END.search(line))
        non_space = [ch for ch in stripped if not ch.isspace()]
        letter_ratio = sum(ch.isalpha() for ch in non_space) / len(non_space)
        if ends_with_page / len(lines) >= 0.6 and letter_ratio >= 0.5:
            return "lines ending in page numbers (contents or index)"

    return None


# True for chunks that carry no explanatory content
def is_noise_chunk(text):
    return noise_reason(text) is not None