import json
import uuid

from langchain_core.messages import HumanMessage
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference
from ragas.run_config import RunConfig

from langgraph_backend import chatbot, llm
from rag_utils import get_retriever_for_thread, ingest_pdf_for_thread
from rag_utils import embeddings as rag_embeddings


test_questions = [
    {
        "question": "What is supervised learning?",
        "ground_truth": "Supervised learning is a type of machine learning where the training data fed to the algorithm includes the desired solutions, called labels."
    },
    {
        "question": "What is the difference between linear regression and logistic regression?",
        "ground_truth": "Linear regression outputs a continuous numerical value directly, while logistic regression passes that value through a sigmoid function to output a probability between 0 and 1, used for classification."
    },
    {
        "question": "What is regularization used for in machine learning?",
        "ground_truth": "Regularization is used to reduce overfitting by constraining a model, typically by reducing the degrees of freedom it has."
    },
]


def run_evaluation(thread_id: str):
    results = []

    for item in test_questions:
        question = item["question"]

        CONFIG = {"configurable": {"thread_id": thread_id}}
        response = chatbot.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=CONFIG
        )
        answer = response["messages"][-1].content
        if isinstance(answer, list):
            answer = " ".join(
                block.get("text", "") for block in answer
                if isinstance(block, dict) and block.get("type") == "text"
            )

        retriever = get_retriever_for_thread(thread_id)
        retrieved_docs = retriever.invoke(question) if retriever else []
        contexts = [doc.page_content for doc in retrieved_docs]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })

        print(f"Q: {question}")
        print(f"A: {answer[:200]}...")
        print(f"Retrieved {len(contexts)} chunks")
        print()

    return results


def score_with_ragas(results):
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in results
    ])

    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(rag_embeddings)

    run_config = RunConfig(
        timeout=180,  # give each judge-LLM call up to 3 minutes
        max_workers=1,  # run evaluations sequentially, not concurrently
    )

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(strictness=1), LLMContextPrecisionWithReference()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
    )

    return scores


if __name__ == "__main__":
    test_thread_id = str(uuid.uuid4())
    pdf_path = "test.pdf"

    print(f"Using thread_id: {test_thread_id}")
    print(f"Ingesting {pdf_path}...")
    summary = ingest_pdf_for_thread(pdf_path, thread_id=test_thread_id, filename=pdf_path)
    print(f"Ingested {summary['chunks']} chunks.\n")

    results = run_evaluation(test_thread_id)

    print("Running RAGAS evaluation...")
    scores = score_with_ragas(results)
    print("\n=== RAGAS Scores ===")
    print(scores)

    with open("evaluation_results.json", "w") as f:
        json.dump({
            "scores": {
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "context_precision": scores["llm_context_precision_with_reference"],
            },
            "details": results,
        }, f, indent=2)

    print("\nSaved results to evaluation_results.json")