import json
import logging
from typing import List, Dict, Tuple, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL_NAME, LLM_TEMPERATURE, TOP_K_RETRIEVAL
from ingestion.vector_store_manager import VectorStoreManager
from .graph_state import GraphState
from .prompts import (
    PREPROCESS_QUERY_PROMPT_TEMPLATE,
    GENERATE_ANSWER_FROM_DOCS_PROMPT_TEMPLATE,
    COMPARE_AND_SYNTHESIZE_PROMPT_TEMPLATE,
    CHECK_GENERAL_KNOWLEDGE_NECESSITY_PROMPT_TEMPLATE
)

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, convert_system_message_to_human=True)
vsm = VectorStoreManager()

def format_chat_history_for_prompt(chat_history: List[BaseMessage], max_turns=5) -> str:
    if not chat_history:
        return "No chat history available."
    history_str = []
    recent_history = chat_history[-(max_turns * 2):]
    for msg in recent_history:
        if isinstance(msg, HumanMessage):
            history_str.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_str.append(f"AI: {msg.content}")
    return "\n".join(history_str)

def format_docs_for_prompt(retrieved_docs: List[Dict]) -> str:
    if not retrieved_docs:
        return "No documents retrieved to format."
    formatted = []
    for i, doc_data in enumerate(retrieved_docs): # Use doc_data to avoid conflict
        meta = doc_data.get('metadata', {})
        # Using the full content for the LLM prompt as snippet might be too short
        content_for_llm = doc_data.get('content', 'Error: Content missing in retrieved document.')
        formatted.append(
            f"Document {i+1} (Source: {meta.get('source_file', 'N/A')}, Page: {meta.get('page_number', 'N/A')}, Type: {meta.get('chunk_type', 'text')}):\n"
            f"{content_for_llm}\n---END DOCUMENT {i+1}---\n" # Clearer end marker
        )
    return "\n".join(formatted)

def preprocess_query_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Preprocessing Query ---")
    user_query = state["original_query"]
    chat_history = state["chat_history"]
    prompt = ChatPromptTemplate.from_template(PREPROCESS_QUERY_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    formatted_hist = format_chat_history_for_prompt(chat_history)
    try:
        response_str = chain.invoke({"chat_history_formatted": formatted_hist, "user_query": user_query})
        logger.debug(f"Preprocess LLM response string: {response_str}")
        if response_str.strip().startswith("```json"):
            response_str = response_str.strip()[7:-3].strip()
        elif response_str.strip().startswith("```"):
            response_str = response_str.strip()[3:-3].strip()
        parsed_response = json.loads(response_str)
        logger.info(f"Parsed preprocess response: {parsed_response}")

        state["processed_query"] = parsed_response.get("retrieval_query", user_query)
        state["use_external_knowledge_explicitly_forbidden"] = parsed_response.get("external_knowledge_forbidden", False)
        
        explicit_files_from_llm = parsed_response.get("explicit_filenames")
        state["target_files_explicit"] = explicit_files_from_llm

        if explicit_files_from_llm:
            normalized_query_filenames = [f_name.lower() for f_name in explicit_files_from_llm]
            state["retrieval_filter"] = {"normalized_filter_filename": {"$in": normalized_query_filenames}}
            logger.info(f"Setting retrieval filter for normalized filenames: {state['retrieval_filter']}")
        else:
            state["retrieval_filter"] = None
        
        value_query_intent = parsed_response.get("value_query_intent", {})
        if state["target_files_explicit"] or value_query_intent.get("is_value_query", False):
            state["query_requires_doc_search"] = True
            # If it's a value query but no specific file, ensure filter is None to search all
            if value_query_intent.get("is_value_query", False) and not state["target_files_explicit"]:
                state["retrieval_filter"] = None 
                logger.info("Value query without specific file: clearing retrieval filter to search all docs.")
        else:
            state["query_requires_doc_search"] = False
    except Exception as e:
        logger.error(f"Error in preprocess_query_node: {e}", exc_info=True)
        state["processed_query"] = user_query
        state["error_message"] = f"Error during query preprocessing: {str(e)[:100]}"
    return state

def retrieve_documents_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Retrieving Documents ---")
    should_search = state.get("query_requires_doc_search", False)
    # Ensure search happens if files are explicitly targeted, even if query_requires_doc_search was False
    if state.get("target_files_explicit") and not state.get("retrieval_filter"):
        # This case can happen if LLM extracts filename but the query itself wasn't deemed "value_query"
        # We still need to build the filter based on target_files_explicit if it's not already set
        normalized_query_filenames = [f_name.lower() for f_name in state["target_files_explicit"]]
        state["retrieval_filter"] = {"normalized_filter_filename": {"$in": normalized_query_filenames}}
        logger.info(f"Re-setting retrieval filter based on target_files_explicit: {state['retrieval_filter']}")
    
    if state.get("target_files_explicit"): # If specific files are mentioned, we must search.
        should_search = True

    if not should_search:
        logger.info("Skipping document retrieval as not explicitly required by query type or file mentions.")
        state["retrieved_docs"] = []
        state["doc_search_performed"] = False
        return state

    query_text = state["processed_query"]
    filter_metadata = state.get("retrieval_filter")
    
    logger.info(f"Retrieving documents for query: '{query_text}' with filter: {filter_metadata}")
    try:
        retrieved = vsm.query_store(
            query_text=query_text,
            n_results=TOP_K_RETRIEVAL,
            filter_metadata=filter_metadata
        )
        state["retrieved_docs"] = retrieved
        state["doc_search_performed"] = True
        logger.info(f"Retrieved {len(retrieved)} documents.") # Standard log
        if retrieved:
            for i, doc_data in enumerate(retrieved):
                 meta = doc_data.get('metadata',{})
                 content_preview = doc_data.get('content', '') # Get full content for debug log
                 # Log more info, including chunk type and a longer preview
                 logger.info( 
                     f"  Retrieved Doc {i+1}: {meta.get('source_file')} pg {meta.get('page_number')} "
                     f"(Type: {meta.get('chunk_type')}, NormFile: {meta.get('normalized_filter_filename')}, HeuristicTitle: {meta.get('table_title_heuristic','N/A')}, Dist: {doc_data.get('distance', -1):.4f})"
                     f"\n    Content Preview (first 200 chars): {content_preview[:200]}..."
                 )
        elif filter_metadata: # If filter was applied but nothing found
            logger.warning(f"No documents retrieved despite filter: {filter_metadata}. Ensure filename metadata in DB matches normalized query filenames and content is relevant.")
        else: # No filter applied and nothing found
            logger.warning("No documents retrieved and no filter was applied. Check query relevance to vector store content.")
    except Exception as e:
        logger.error(f"Error in retrieve_documents_node: {e}", exc_info=True)
        state["retrieved_docs"] = []
        state["doc_search_performed"] = False
        state["error_message"] = f"Error retrieving documents: {str(e)[:100]}"
    return state

# --- Other nodes (generate_answer_from_docs_node, decide_general_knowledge_route_node, etc.) ---
# --- remain the same as the last complete version you had, ensure BaseMessage is imported ---
# --- Make sure the format_docs_for_prompt uses full content for LLM, not just snippet ---

def generate_answer_from_docs_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Generating Answer from Documents ---")
    doc_search_performed = state.get("doc_search_performed", False)
    docs_were_expected = state.get("query_requires_doc_search", False) or bool(state.get("target_files_explicit"))

    if not state.get("retrieved_docs"): # Check if any docs were retrieved
        if doc_search_performed and docs_were_expected:
            logger.warning("No documents were retrieved, although a document search was performed and expected. Cannot generate answer from docs.")
            state["answer_from_docs_only"] = "No relevant documents were found to answer your query based on the specified criteria."
        else: # e.g., search not required, or search failed to retrieve
            logger.info("No documents available to generate answer from.")
            state["answer_from_docs_only"] = None 
        state["citations"] = []
        return state

    user_query = state["original_query"] 
    chat_history = state["chat_history"]
    retrieved_docs = state["retrieved_docs"]

    formatted_hist = format_chat_history_for_prompt(chat_history)
    # Critical: Ensure format_docs_for_prompt sends enough (or full) content to LLM
    formatted_docs = format_docs_for_prompt(retrieved_docs) 

    prompt = ChatPromptTemplate.from_template(GENERATE_ANSWER_FROM_DOCS_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "user_query": user_query,
            "chat_history_formatted": formatted_hist,
            "formatted_document_context": formatted_docs
        })
        
        citations_found = []
        import re
        citation_pattern = r"\[Source:\s*(?P<file>[^,]+),\s*Page:\s*(?P<page>\d+)\s*]"
        for match in re.finditer(citation_pattern, answer):
            citations_found.append({
                "source_file": match.group("file").strip(),
                "page_number": int(match.group("page")),
            })
        
        unique_citations = []
        seen_citations = set()
        for cit in citations_found:
            cit_tuple = (cit["source_file"], cit["page_number"])
            if cit_tuple not in seen_citations:
                unique_citations.append(cit)
                seen_citations.add(cit_tuple)

        state["answer_from_docs_only"] = answer
        state["citations"] = unique_citations
        logger.info(f"Generated answer from docs. Citations found: {len(unique_citations)}")

    except Exception as e:
        logger.error(f"Error in generate_answer_from_docs_node: {e}", exc_info=True)
        state["answer_from_docs_only"] = "Sorry, an error occurred while generating the answer from documents."
        state["citations"] = []
        state["error_message"] = f"Error generating answer from docs: {str(e)[:100]}"
    return state


def decide_general_knowledge_route_node(state: GraphState) -> str:
    logger.info("--- ROUTER: Decide General Knowledge Route ---")
    
    if state.get("error_message"): 
        logger.warning(f"Error detected: {state['error_message']}. Routing to handle_error.")
        return "handle_error"

    if state.get("use_external_knowledge_explicitly_forbidden"):
        logger.info("External knowledge explicitly forbidden. Routing to format final response.")
        if state.get("answer_from_docs_only"):
            state["final_response_for_user"] = state["answer_from_docs_only"]
        elif state.get("doc_search_performed") and (state.get("query_requires_doc_search") or state.get("target_files_explicit")):
             state["final_response_for_user"] = "I couldn't find relevant information in the specified documents to answer your query."
        else: 
             state["final_response_for_user"] = "I cannot answer this query with the current restrictions as no relevant documents were found or searched."
        return "format_final_response"

    doc_ans = state.get("answer_from_docs_only")
    # Check if doc search was performed AND expected to yield results
    doc_search_performed_and_expected = state.get("doc_search_performed") and \
                                        (state.get("query_requires_doc_search") or bool(state.get("target_files_explicit")))

    # If doc search was expected & performed but yielded no conclusive answer or no docs retrieved
    # A "conclusive answer" should not contain phrases indicating failure.
    no_conclusive_doc_answer = not doc_ans or \
                               any(phrase in doc_ans.lower() for phrase in 
                                   ["cannot answer", "not contain sufficient information", "no relevant documents", "couldn't find relevant information"])

    if doc_search_performed_and_expected and no_conclusive_doc_answer:
        logger.info("Document search expected and performed but yielded no conclusive answer. Routing to generate general knowledge answer.")
        state["should_compare_with_general_knowledge"] = False 
        return "generate_general_knowledge"

    # If doc search was not performed (e.g. general query from start) and no doc answer
    if not state.get("doc_search_performed") and not doc_ans:
        logger.info("No document search performed and no document answer. Routing to generate general knowledge answer.")
        state["should_compare_with_general_knowledge"] = False
        return "generate_general_knowledge"
        
    if doc_ans and not no_conclusive_doc_answer: # We have a conclusive document answer
        logger.info("Conclusive document answer exists. Checking if general knowledge is needed for enhancement.")
        try:
            prompt = ChatPromptTemplate.from_template(CHECK_GENERAL_KNOWLEDGE_NECESSITY_PROMPT_TEMPLATE)
            chain = prompt | llm | StrOutputParser()
            response_str = chain.invoke({
                "user_query": state["original_query"],
                "answer_from_docs": doc_ans
            })
            if response_str.strip().startswith("```json"):
                response_str = response_str.strip()[7:-3].strip()
            elif response_str.strip().startswith("```"):
                response_str = response_str.strip()[3:-3].strip()
            parsed_check = json.loads(response_str)

            if parsed_check.get("needs_general_knowledge", False):
                logger.info(f"LLM decided general knowledge is needed. Reason: {parsed_check.get('reason')}")
                state["should_compare_with_general_knowledge"] = True
                return "generate_general_knowledge"
            else:
                logger.info("LLM decided general knowledge is NOT needed. Using docs answer as final.")
                state["final_response_for_user"] = doc_ans
                return "format_final_response"
        except Exception as e:
            logger.error(f"Error in LLM check for general knowledge necessity: {e}. Defaulting to comparison (if doc_ans exists).")
            state["should_compare_with_general_knowledge"] = True 
            return "generate_general_knowledge"
            
    logger.info("Fallback in router. Using document answer as final if available, else try general.")
    if doc_ans: # Even if not deemed "conclusive", it's what we have from docs
        state["final_response_for_user"] = doc_ans
        return "format_final_response"
    else: 
        state["should_compare_with_general_knowledge"] = False
        return "generate_general_knowledge"


def generate_general_knowledge_answer_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Generating General Knowledge Answer ---")
    user_query = state["original_query"] 
    chat_history = state["chat_history"]
    formatted_hist = format_chat_history_for_prompt(chat_history)
    prompt_str = f"Chat History (for context, if any):\n{formatted_hist}\n\nUser Query: {user_query}\n\nBased on your general knowledge, please provide an answer to the user's query."
    try:
        response = llm.invoke([HumanMessage(content=prompt_str)])
        state["answer_from_general_knowledge"] = response.content
        logger.info("Generated general knowledge answer.")
    except Exception as e:
        logger.error(f"Error in generate_general_knowledge_answer_node: {e}", exc_info=True)
        state["answer_from_general_knowledge"] = "Sorry, an error occurred while trying to answer from general knowledge."
        state["error_message"] = f"Error generating general knowledge: {str(e)[:100]}"
    return state

def synthesize_answers_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Synthesizing Answers ---")
    answer_docs = state.get("answer_from_docs_only")
    answer_general = state.get("answer_from_general_knowledge")
    citations = state.get("citations", [])
    should_compare = state.get("should_compare_with_general_knowledge", False) # Check if comparison was intended

    # If comparison was not intended, or only one answer exists, choose appropriately
    if not should_compare or not (answer_docs and answer_general):
        if answer_docs and not any(phrase in answer_docs.lower() for phrase in ["cannot answer", "not contain sufficient information", "no relevant documents"]):
            logger.info("Using document answer as final (no synthesis needed or only docs answer is valid).")
            state["final_response_for_user"] = answer_docs
            # Citations already set
        elif answer_general:
            logger.info("Using general knowledge answer as final (no synthesis needed or only general answer is valid).")
            state["final_response_for_user"] = answer_general
            state["citations"] = [] # Clear citations if only general knowledge is used
        else: # Neither is conclusive or available
            logger.info("Neither document nor general knowledge answer is conclusive for final response.")
            state["final_response_for_user"] = state.get("final_response_for_user") or \
                                            answer_docs or \
                                            "I could not find a conclusive answer from available sources."
        return state

    # Proceed with synthesis as both answers exist and comparison was intended
    logger.info("Proceeding with synthesis as both answers exist and comparison was intended.")
    prompt = ChatPromptTemplate.from_template(COMPARE_AND_SYNTHESIZE_PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()
    try:
        formatted_citations = json.dumps(citations) 
        synthesized = chain.invoke({
            "user_query": state["original_query"],
            "answer_from_docs": answer_docs,
            "citations_from_docs": formatted_citations,
            "answer_from_general": answer_general
        })
        state["synthesized_answer"] = synthesized
        state["final_response_for_user"] = synthesized 
        # Citations from the original doc answer are usually kept, synthesis prompt guides LLM.
        logger.info("Synthesized answers successfully.")
    except Exception as e:
        logger.error(f"Error in synthesize_answers_node: {e}", exc_info=True)
        state["final_response_for_user"] = f"Sorry, an error occurred while combining information. From documents: {answer_docs or 'N/A'}. From general knowledge: {answer_general or 'N/A'}."
        state["error_message"] = f"Error synthesizing answers: {str(e)[:100]}"
    return state

def format_final_response_node(state: GraphState) -> GraphState:
    logger.info("--- NODE: Formatting Final Response ---")
    # Prioritize error message if one was set and not already part of the response
    if state.get("error_message") and not (state.get("final_response_for_user", "").lower().startswith("an error occurred") or state.get("final_response_for_user", "").lower().startswith("i apologize")):
        state["final_response_for_user"] = f"An error occurred: {state['error_message']}"
        state["citations"] = [] # Clear citations if there was an error leading to this
    elif not state.get("final_response_for_user"): # Fallback if no response set by any prior logic
        # This tries to pick the most complete answer available
        if state.get("synthesized_answer"):
            state["final_response_for_user"] = state["synthesized_answer"]
        elif state.get("answer_from_docs_only") and not any(phrase in state["answer_from_docs_only"].lower() for phrase in ["cannot answer", "not contain sufficient information", "no relevant documents"]):
            state["final_response_for_user"] = state["answer_from_docs_only"]
        elif state.get("answer_from_general_knowledge"):
            state["final_response_for_user"] = state["answer_from_general_knowledge"]
            state["citations"] = [] # No doc citations for general knowledge
        elif state.get("answer_from_docs_only"): # Fallback to even a negative doc answer
             state["final_response_for_user"] = state["answer_from_docs_only"]
        else:
            state["final_response_for_user"] = "I'm sorry, I was unable to process your request fully or find a conclusive answer."
            state["citations"] = []
    
    logger.info(f"Final response for user: {state['final_response_for_user'][:200]}...")
    if state.get("citations"):
        logger.info(f"Final citations: {state['citations']}")
    else:
        logger.info("No citations for the final response.")
    return state

def handle_error_node(state: GraphState) -> GraphState:
    error_msg = state.get('error_message', 'An unknown error occurred during processing.')
    logger.error(f"--- NODE: Error Handler --- Error: {error_msg}")
    state["final_response_for_user"] = f"I apologize, an error occurred: {error_msg}"
    state["citations"] = [] 
    return state