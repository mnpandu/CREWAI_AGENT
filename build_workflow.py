from graph_state import GraphState
from graph_nodes import (
    retrieve,
    grade_documents,
    generate,
    transform_query,
    decide_to_generate,
)
from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate"
    },
)
workflow.add_edge("transform_query", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()
if __name__ == "__main__":
    app.visualize("workflow_graph.html")