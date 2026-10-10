import glob
import json
import sys

from pypdf import PdfReader


# lowercases and collapses all whitespace so line breaks in the PDF text don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# checks that every question's evidence phrase(s) really appear in its PDF's extracted text; folder defaults to eval_questions
def main(folder="evals/questions/standard"):
    pdf_text_cache = {}
    missing = 0

    for path in sorted(glob.glob(f"{folder}/*.json")):
        with open(path) as f:
            items = json.load(f)

        for item in items:
            doc = item["doc"]
            if doc not in pdf_text_cache:
                reader = PdfReader(doc)
                pages = [(page.extract_text() or "") for page in reader.pages]
                pdf_text_cache[doc] = normalize(" ".join(pages))

            # evidence may be one phrase or a list (multi-passage); unanswerable questions have an empty list and are skipped
            phrases = item["evidence"] if isinstance(item["evidence"], list) else [item["evidence"]]
            found = all(normalize(phrase) in pdf_text_cache[doc] for phrase in phrases)
            print("OK     " if found else "MISSING", doc, "|", item["question"][:60])
            if not found:
                missing += 1

    print(f"\n{missing} question(s) with missing evidence")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evals/questions/standard")