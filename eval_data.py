print("Starting evaluation...")

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from rag_chain import build_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time
import os 

load_dotenv()

TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "What are the key points mentioned?",
    "Who is this document about?"
]



def run_evaluation():
    chain, retriever = build_qa_chain()

    questions, answers, contexts = [], [], []

    for q in TEST_QUESTIONS:
        print(f"Asking: {q}")
        answer = chain.invoke(q)
        source_docs = retriever.invoke(q)
        questions.append(q)
        answers.append(answer)
        contexts.append([doc.page_content for doc in source_docs])
        time.sleep(3)

    print("\n--- DEBUG ---")
    for i in range(len(questions)):
        print(f"Q: {questions[i]}")
        print(f"A: {answers[i]}")
        print(f"Contexts: {contexts[i][:1]}")  # just first context to keep it short
        print()

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    })

    

    gemini_llm = LangchainLLMWrapper(
    ChatGoogleGenerativeAI(
        model="models/gemini-3.1-flash-lite",
        google_api_key=os.environ["GOOGLE_API_KEY"],
        generation_config={"temperature": 0}
    )
)
    gemini_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    faithfulness_metric = faithfulness
    faithfulness_metric.llm = gemini_llm

    relevancy_metric = answer_relevancy
    relevancy_metric.llm = gemini_llm
    relevancy_metric.embeddings = gemini_embeddings

    scores = evaluate(
        dataset,
        metrics=[faithfulness_metric, relevancy_metric],
        llm=gemini_llm,
        embeddings=gemini_embeddings,
        raise_exceptions=False
    )

    print("\n=== RAGAS Evaluation Results ===")
    print(f"Faithfulness:      {scores['faithfulness']}")
    print(f"Answer Relevancy:  {scores['answer_relevancy']}")
    print("================================")
    print("Screenshot this and add to your README + resume!")
    return scores

if __name__ == "__main__":
    run_evaluation()