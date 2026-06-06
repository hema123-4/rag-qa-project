from langgraph.graph import StateGraph, END
from state import RAGState
from nodes import retrieve_node, generate_node, critic_node, reformulate_node, finalize_node, should_retry

def build_self_healing_rag(retriever):
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node(retriever))
    graph.add_node("generate", generate_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")
    graph.add_conditional_edges("critic", should_retry, {
        "finalize": "finalize",
        "reformulate": "reformulate"
    })
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("finalize", END)

    return graph.compile()