import glob
import json

from pypdf import PdfReader


# lowercases and collapses all whitespace so line breaks in the PDF text don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# checks that every question's evidence phrase really appears in its PDF's extracted text
def main():
    pdf_text_cache = {}
    missing = 0

    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            items = json.load(f)

        for item in items:
            doc = item["doc"]
            if doc not in pdf_text_cache:
                reader = PdfReader(doc)
                pages = [(page.extract_text() or "") for page in reader.pages]
                pdf_text_cache[doc] = normalize(" ".join(pages))

            found = normalize(item["evidence"]) in pdf_text_cache[doc]
            print("OK     " if found else "MISSING", doc, "|", item["question"][:60])
            if not found:
                missing += 1

    print(f"\n{missing} question(s) with missing evidence")


if __name__ == "__main__":
    main()