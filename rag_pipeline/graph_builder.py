from langgraph.graph import StateGraph, END
from .graph_state import GraphState
from .nodes import (
    preprocess_query_node,
    retrieve_documents_node,
    generate_answer_from_docs_node,
    decide_general_knowledge_route_node,
    generate_general_knowledge_answer_node,
    synthesize_answers_node,
    format_final_response_node,
    handle_error_node
)
import logging

logger = logging.getLogger(__name__)

def build_rag_graph():
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("preprocess_query", preprocess_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_answer_from_docs", generate_answer_from_docs_node)
    workflow.add_node("generate_general_knowledge", generate_general_knowledge_answer_node)
    workflow.add_node("synthesize_answers", synthesize_answers_node)
    workflow.add_node("format_final_response", format_final_response_node)
    workflow.add_node("handle_error", handle_error_node) # Error handling node

    # Set entry point
    workflow.set_entry_point("preprocess_query")

    # Define edges
    workflow.add_edge("preprocess_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer_from_docs")
    
    # Conditional routing after attempting to generate answer from docs
    workflow.add_conditional_edges(
        "generate_answer_from_docs",
        decide_general_knowledge_route_node, # This router function returns the next node name
        {
            "generate_general_knowledge": "generate_general_knowledge",
            "format_final_response": "format_final_response", # Directly to formatting if no general knowledge needed
            "end_error": "handle_error" # Route to error handler
        }
    )

    workflow.add_conditional_edges(
        "generate_general_knowledge",
        # Simple decision: if we generated general knowledge, do we have docs answer to synthesize with?
        lambda state: "synthesize_answers" if state.get("answer_from_docs_only") and state.get("should_compare_with_general_knowledge") else "format_final_response",
        {
            "synthesize_answers": "synthesize_answers",
            "format_final_response": "format_final_response" # If no doc answer, or no comparison flag, general becomes final
        }
    )
    
    workflow.add_edge("synthesize_answers", "format_final_response")
    workflow.add_edge("format_final_response", END) # End of successful execution
    workflow.add_edge("handle_error", END) # End after error handling

    # Compile the graph
    app = workflow.compile()
    logger.info("RAG Graph compiled successfully.")
    return app

# Example of how to get the graph (can be called from main.py)
rag_app = build_rag_graph()

if __name__ == '__main__':
    # To visualize the graph (optional, requires graphviz)
    try:
        from PIL import Image
        import io
        img_bytes = rag_app.get_graph().draw_mermaid_png()
        if img_bytes:
            with open("rag_graph.png", "wb") as f:
                f.write(img_bytes)
            print("Graph visualization saved to rag_graph.png (Mermaid format PNG)")
        else:
            print("Could not generate graph image (is graphviz installed and in PATH?).")
    except ImportError:
        print("Pillow or other imaging library not installed, skipping graph visualization.")
    except Exception as e:
        print(f"Error generating graph image: {e}")