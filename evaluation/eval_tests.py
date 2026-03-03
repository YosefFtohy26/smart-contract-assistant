from app.chain import get_conversational_rag_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


# -----------------------------------
# 1️⃣ Golden Set (Example)
# -----------------------------------
eval_questions = [
    {
        "question": "What is the termination period for this contract?",
        "ground_truth": "The contract can be terminated with 30 days written notice."
    },
    {
        "question": "Who are the primary parties involved?",
        "ground_truth": "The parties are TechCorp Inc. and Global Services Ltd."
    }
]


# -----------------------------------
# 2️⃣ Semantic Similarity Metric
# -----------------------------------
def compute_similarity(answer, ground_truth, embeddings):

    emb_answer = embeddings.embed_query(answer)
    emb_truth = embeddings.embed_query(ground_truth)

    score = cosine_similarity([emb_answer], [emb_truth])[0][0]

    return round(float(score), 3)


# -----------------------------------
# 3️⃣ Evaluation Runner
# -----------------------------------
def run_evaluation():

    chain = get_conversational_rag_chain()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    results = []

    print("🧪 Starting Evaluation...\n")

    for item in eval_questions:

        response = chain.invoke({"input": item["question"]})

        system_answer = response["answer"]

        similarity_score = compute_similarity(
            system_answer,
            item["ground_truth"],
            embeddings
        )

        results.append({
            "Question": item["question"],
            "Ground Truth": item["ground_truth"],
            "System Answer": system_answer,
            "Semantic Similarity Score": similarity_score
        })

        print(f"Question: {item['question']}")
        print(f"Similarity: {similarity_score}\n")

    df = pd.DataFrame(results)
    df.to_csv("evaluation/eval_report.csv", index=False)

    print("✅ Evaluation complete.")
    print("Results saved to evaluation/eval_report.csv")

    return df


if __name__ == "__main__":
    run_evaluation()