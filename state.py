from typing import TypedDict

class RAGState(TypedDict):
    question: str
    reformulated_question: str
    context: str
    answer: str
    is_grounded: bool
    retry_count: int
    final_answer: str