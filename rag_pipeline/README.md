
# RAG Pipeline Module

This module contains the core logic for the Retrieval Augmented Generation (RAG) pipeline. It uses LangChain and LangGraphs to define a stateful workflow that processes user queries, retrieves relevant information from the vector store, and generates answers using a Large Language Model (LLM).

## Overview

The RAG pipeline, orchestrated as a LangGraph, typically involves these stages:

1.  **Query Preprocessing:** The user's query might be rephrased, expanded, or classified to improve retrieval or determine its intent (e.g., if it's a general question or refers to specific documents). This step often involves an LLM call.
2.  **Retrieval:** Based on the (preprocessed) query, relevant document chunks are retrieved from the ChromaDB vector store managed by `VectorStoreManager` from the `ingestion` module.
3.  **Context Augmentation & Formatting:** The retrieved chunks are formatted and combined to form a coherent context string.
4.  **Answer Generation:** The original query and the augmented context are passed to an LLM (e.g., Google Gemini) to generate a final answer.
5.  **State Management:** LangGraphs is used to manage the flow between these stages, handle conversational history, and allow for more complex interactions like follow-up questions or conditional routing based on intermediate results (e.g., whether documents were retrieved).

## Key Files

*   `app.py` (or `main.py`, `cli.py`): The main entry point for the RAG application. This could be a command-line interface, a Streamlit web application, or a FastAPI backend. It initializes and runs the LangGraph.
*   `graph.py`: Defines the LangGraph state machine. This includes:
    *   The state schema (e.g., `State` class inheriting from `TypedDict`).
    *   Node definitions (functions or callables from `nodes.py`).
    *   Edge definitions connecting the nodes, including conditional edges.
    *   Compilation of the graph into a runnable `CompiledGraph`.
*   `nodes.py`: Contains the implementation of individual nodes that perform specific tasks within the LangGraph. Examples:
    *   `preprocess_query_node`: Modifies or analyzes the input query.
    *   `retrieve_documents_node`: Interacts with the vector store to get relevant chunks.
    *   `format_context_node`: Prepares retrieved documents for the LLM.
    *   `generate_answer_node`: Calls the LLM to generate the final response.
    *   `should_retrieve_node` (or similar): A conditional node deciding the next step.
*   `(config.py)`: (Optional) May contain configuration settings specific to the RAG pipeline, such as LLM model names (e.g., `gemini-1.5-pro`), prompt templates, retrieval parameters (e.g., number of documents to retrieve `k`), or temperature settings for the LLM.

## Workflow (LangGraph-based)

1.  The `app.py` initializes the RAG system, including loading the vector store retriever and compiling the LangGraph defined in `graph.py`.
2.  When a user submits a query, it's passed as input to the LangGraph's `stream` or `invoke` method.
3.  The graph execution begins, traversing nodes based on the defined edges and conditions. For instance:
    *   A query preprocessing node might refine the query.
    *   A conditional node might decide if retrieval is needed.
    *   The retrieval node queries ChromaDB.
    *   A formatting node prepares the context.
    *   The generation node calls Gemini with the query and context.
4.  The graph manages conversational history by updating its state between interactions if designed to do so.
5.  The final output (e.g., the generated answer) is returned by the graph.

## Configuration

*   **LLM Settings:** The LLM (e.g., Google Gemini) used for various nodes is typically configured in `nodes.py` or a shared config, referencing the API key from the `.env` file.
*   **Prompt Templates:** Prompts are crucial and are usually defined as `ChatPromptTemplate` or `PromptTemplate` instances within `nodes.py` or a dedicated prompts module.
*   **Retriever Settings:** The vector store retriever (from `VectorStoreManager`) is initialized with parameters like `k` (number of documents) and search type. This is typically done in `app.py` and passed to relevant nodes, or the retriever is accessed directly within a node.
*   **LangGraph Structure:** The core flow, state variables, and conditional logic are all defined within `graph.py`.

## How to Run

Assuming `app.py` is the entry point for your interactive RAG application:

Navigate to the project's root directory and ensure your virtual environment is activated.

**For a CLI application:**
```bash
python -m rag_pipeline.app