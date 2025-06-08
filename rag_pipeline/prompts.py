# Prompts for various LLM calls in the RAG pipeline

PREPROCESS_QUERY_PROMPT_TEMPLATE = """
Given the user's query and the chat history, perform the following tasks:

1.  **Identify Explicit Filenames:** List any filenames (e.g., "paper1.pdf", "report_final.docx") explicitly mentioned by the user in their LATEST query. If a partial name is given (e.g., "iso 23277"), try to complete it if it's clear (e.g., "iso 23277.pdf").
2.  **Identify Target Table Name/Number/Description:** If the user's LATEST query explicitly mentions a table by its name, number, or a descriptive title (e.g., "Table A.1", "Table 3", "the table about testing parameters", "Figure 2", "Recommended testing parameters table"), extract this table identifier as accurately as possible. If no specific table identifier is mentioned, this should be null.
3.  **External Knowledge Restriction:** Determine if the user's LATEST query explicitly forbids using external/general knowledge (e.g., "only from the PDF", "don't use the web").
4.  **Core Question for Retrieval:** Rephrase the user's LATEST query into a concise question or statement optimized for semantic search in a document vector store.
    *   If a `target_table_identifier` was extracted in step 2, **ensure this exact identifier is included in the rephrased retrieval query**. For example, if query is "explain Table A.1 about testing parameters", the retrieval query should be something like "Table A.1 recommended testing parameters from [filename if provided]". The goal is to make the retrieval query highly similar to the content of the chunk containing that specific table.
    *   If no specific table is mentioned, but the query is generally about tables (e.g., "show me tables"), the retrieval query should reflect that (e.g., "data tables content from [filename if provided]").
    *   Incorporate relevant context from chat history if it clarifies the current query.
5.  **Chat Coherence & Memory:** Identify if the user's LATEST query refers to information they provided earlier in the chat history.
6.  **Value/Table Query Intent:** Determine if the query is primarily asking for:
    *   A specific numerical value or range.
    *   An explanation of a table (general or specific).
    *   Variables, units, or specific data points from a document.

Chat History (most recent first):
{chat_history_formatted}

User's Latest Query: {user_query}

Output your response as a JSON object with the following schema:
{{
    "explicit_filenames": ["filename1.pdf"] | null,
    "target_table_identifier": "Table A.1 — Recommended testing parameters" | "Table 3" | "the table about results" | null,
    "external_knowledge_forbidden": boolean,
    "retrieval_query": "Rephrased question for vector search including table identifier if present...",
    "references_chat_memory": {{
        "is_referenced": boolean,
        "details": "Brief explanation if memory is referenced, e.g., 'User refers to their name mentioned earlier.'"
    }},
    "value_query_intent": {{
        "is_value_query": boolean,
        "type": "specific_value" | "table_explanation" | "data_points" | "none",
        "details": "e.g., 'User asks for the value of X in section Y', 'User wants Table A.1 explained'"
    }}
}}
"""


GENERATE_ANSWER_FROM_DOCS_PROMPT_TEMPLATE = """
You are a helpful AI assistant for Civil Engineering research.
Answer the user's query based *ONLY* on the provided context from research papers.
If the information is not in the provided context, clearly state that you cannot answer from the given documents or that the specific information (e.g., a named table) was not found in the retrieved sections.
Do NOT use any external or general knowledge unless explicitly asked to compare later.

User Query: {user_query}

Chat History (for context, if relevant to the query):
{chat_history_formatted}

Provided Context from Documents:
---BEGIN DOCUMENT CONTEXT---
{formatted_document_context}
---END DOCUMENT CONTEXT---

Task:
1.  Carefully read the User Query and the Provided Context.
2.  If the query asks to explain a specific named table (e.g., "Table A.1"):
    *   Scan the provided context for a chunk that represents this table. Look for titles or content matching the table name.
    *   If found, explain the contents of that specific table comprehensively. Describe its columns, rows, and the data it presents.
    *   If the specific named table is NOT found in the provided context, clearly state "The specific table '[table name from query]' was not found in the retrieved document sections."
3.  If the query asks for specific values, data points, or ranges (not tied to a specific named table):
    *   Locate the precise information in the context.
    *   Extract exact values, units, and relevant descriptions.
    *   If the information is from a table (even an unnamed one in the context), mention that.
4.  If the query is a general question about tables in the document, describe any tables present in the context.
5.  Construct a comprehensive answer to the User Query.
6.  For every piece of information taken from the documents to form your answer, you MUST provide a citation.
    A citation should look like: [Source: <source_file>, Page: <page_number>]
    If multiple sources/pages contribute to a single point, list all relevant citations.
7.  If the query cannot be answered from the provided context, respond with "The provided documents do not contain sufficient information to answer this query." or a more specific message if a named item wasn't found.

Answer:
"""

COMPARE_AND_SYNTHESIZE_PROMPT_TEMPLATE = """
You are an AI assistant tasked with synthesizing information.
User's Original Query: {user_query}

I have two answers for this query:
1.  Answer based *ONLY* on provided documents:
    "{answer_from_docs}"
    Citations for this answer: {citations_from_docs}

2.  Answer based on general knowledge (or if documents were not conclusive):
    "{answer_from_general}"

Your Task:
Combine these answers into a single, coherent response for the user. Prioritize information from the documents if it's relevant and directly answers the query.

Guidelines:
- If the document-based answer is comprehensive and directly addresses the query (and doesn't state information was not found), use it as the primary response. Ensure all its citations are included. You can then state if general knowledge confirms or adds minor context.
- If general knowledge provides significant information *not* found in the documents but relevant to the query, integrate it. Clearly distinguish which information comes from documents (with citations) and which is general knowledge.
- If the document answer and general knowledge answer are very similar, prefer the document answer with its citations and mention that general knowledge concurs.
- If the query was *explicitly* about information within the documents (e.g., "In paper_X.pdf, what is Table A.1..."), and the document answer is "Information/Table not found", then the final answer should reflect that. You can then say: "However, based on general knowledge, [general knowledge answer if applicable to the broader topic]."
- If there's a direct contradiction between the document and widely accepted general knowledge (and the query wasn't strictly limited to the document), present both and highlight the discrepancy respectfully.

Synthesized Answer for the User:
"""

CHECK_GENERAL_KNOWLEDGE_NECESSITY_PROMPT_TEMPLATE = """
User Query: {user_query}
Answer derived from documents: {answer_from_docs}

Considering the user's query and the answer derived *only* from the provided documents:
Is the answer from documents sufficient and comprehensive?
Or, would incorporating general knowledge significantly enhance the answer by providing broader context, alternative perspectives, or filling critical gaps *not covered by the documents* (assuming the user hasn't forbidden external knowledge, and the document answer wasn't a definitive "information not found for specific request")?

Respond with a JSON object:
{{
    "needs_general_knowledge": boolean,
    "reason": "Brief explanation if true (e.g., 'Docs provide specifics, general knowledge can give broader theory', or 'Docs don't cover aspect X of the query', or 'Document answer stated specific information was not found, general knowledge might address the broader topic.'). If false, state 'Document answer is comprehensive.'"
}}
"""