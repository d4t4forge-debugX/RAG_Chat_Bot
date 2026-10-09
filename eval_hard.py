import argparse
import glob
import json
import os
import time
from datetime import datetime

from evaluation import ingest_documents, ask_chatbot, content_to_text, get_contexts_used_by_chatbot
from langgraph_backend import llm

# seconds to wait between questions so we stay under the free tier's 15 requests per minute
PAUSE_SECONDS = 6


# lowercases and collapses whitespace, same as verify_questions.py, so phrase matching ignores PDF line breaks
def normalize(text):
    return " ".join(text.split()).lower()


# reads every question in <questions_dir>/*.json into one list (default: the hard set)
def load_hard_questions(questions_dir="eval_questions_hard"):
    questions = []
    for path in sorted(glob.glob(f"{questions_dir}/*.json")):
        with open(path) as f:
            questions.extend(json.load(f))
    return questions


# one cheap LLM call: does the answer say the document lacks this information (a refusal) instead of answering?
def is_refusal(question, answer):
    prompt = (
        "You are grading a document question-answering bot. Answer with only YES or NO.\n"
        "Does the bot's answer clearly say that the document does not contain or does not state "
        "the requested information, instead of giving a specific answer?\n\n"
        f"Question: {question}\nBot answer: {answer}"
    )
    reply = content_to_text(llm.invoke(prompt).content)
    return reply.strip().upper().startswith("YES")


# fraction of a question's evidence phrases that appear somewhere in the chunks the bot retrieved
def evidence_coverage(evidence, contexts):
    joined = normalize(" ".join(contexts))
    found = [phrase for phrase in evidence if normalize(phrase) in joined]
    return len(found), len(evidence)


# runs every hard question through the real chatbot and scores it by its type
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="baseline", help="name for this run, used in the output filename")
    parser.add_argument("--questions-dir", default="eval_questions_hard", help="folder of question files to run")
    args = parser.parse_args()

    questions = load_hard_questions(args.questions_dir)
    print(f"Loaded {len(questions)} hard questions.\n")
    base_threads, _ = ingest_documents(questions)
    print()

    records = []
    for i, item in enumerate(questions, start=1):
        print(f"[{i}/{len(questions)}] ({item['type']}) {item['question']}")
        record = {**item, "answer": "", "contexts": [], "status": "ok"}

        response = ask_chatbot(item["question"], base_threads[item["doc"]])
        if response is None:
            record["status"] = "chatbot_error"
        elif "__interrupt__" in response:
            record["status"] = "paused_for_approval"
        else:
            record["answer"] = content_to_text(response["messages"][-1].content).strip()
            record["contexts"] = get_contexts_used_by_chatbot(response["messages"])

        if record["status"] == "ok":
            if item["type"] == "unanswerable":
                record["refused"] = is_refusal(item["question"], record["answer"])
                print(f"   refused correctly: {record['refused']}")
            else:
                found, total = evidence_coverage(item["evidence"], record["contexts"])
                record["evidence_found"] = f"{found}/{total}"
                print(f"   evidence retrieved: {found}/{total}")
        else:
            print(f"   WARNING: status = {record['status']}")

        print(f"   answer: {record['answer'][:200]}\n")
        records.append(record)
        time.sleep(PAUSE_SECONDS)

    unanswerable = [r for r in records if r["type"] == "unanswerable" and r["status"] == "ok"]
    # every answerable question (single-fact or multi-passage) is scored on whether ALL its evidence was retrieved
    multi = [r for r in records if r["type"] != "unanswerable" and r["status"] == "ok"]
    refused = sum(1 for r in unanswerable if r["refused"])
    full_evidence = sum(1 for r in multi if r["evidence_found"].split("/")[0] == r["evidence_found"].split("/")[1])

    print("=== Hard set results ===")
    print(f"Unanswerable refused correctly: {refused}/{len(unanswerable)}")
    print(f"Answerable questions with ALL evidence retrieved: {full_evidence}/{len(multi)}")
    print("Read each multi-passage answer above against its ground truth before trusting these numbers.")

    os.makedirs("eval_runs", exist_ok=True)
    out_path = f"eval_runs/{os.path.basename(args.questions_dir)}_{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump({"refused": f"{refused}/{len(unanswerable)}",
                   "multi_full_evidence": f"{full_evidence}/{len(multi)}",
                   "questions": records}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()