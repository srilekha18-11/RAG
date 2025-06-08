from typing import List, Dict, TypedDict, Optional, Tuple
from langchain_core.messages import BaseMessage # For chat history

class GraphState(TypedDict):
    original_query: str
    processed_query: str 
    chat_history: List[BaseMessage] # Using LangChain's message types
    
    target_files_explicit: Optional[List[str]] # Files explicitly mentioned by user
    inferred_target_files: Optional[List[str]] # Files system infers from query/history
    
    # Retrieval control
    retrieval_filter: Optional[Dict] # Metadata filter for vector store

    retrieved_docs: List[Dict] # List of dicts from vector store: {'content': str, 'metadata': dict}
    
    # Control flow flags
    query_requires_doc_search: bool
    doc_search_performed: bool
    
    use_external_knowledge_explicitly_forbidden: bool # If user says "only from PDFs"
    should_compare_with_general_knowledge: bool 
    
    # Final answer components
    answer_from_docs_only: Optional[str]
    answer_from_general_knowledge: Optional[str]
    synthesized_answer: Optional[str] # If comparison happens

    final_response_for_user: str
    citations: List[Dict] # {'source_file': str, 'page_number': int, 'content_snippet': str}
    
    # For error handling or alternative paths
    error_message: Optional[str]