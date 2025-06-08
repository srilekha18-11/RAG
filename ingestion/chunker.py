import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import logging

logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_parsed_document(self, file_path: str, conference_name: str, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        base_filename = os.path.basename(file_path)
        normalized_filter_filename = base_filename.lower()

        for page_data in parsed_pages:
            page_number = page_data["page_number"]
            text_content = page_data["text_content"]
            # tables_markdown is a list of markdown strings for tables on the page
            tables_markdown_list = page_data.get("tables_markdown", []) # Ensure it's a list
            source_type = page_data["source_type"]

            # Chunk main text content
            if text_content and text_content.strip():
                text_chunks_on_page = self.text_splitter.split_text(text_content)
                for i, chunk_text in enumerate(text_chunks_on_page):
                    chunk_metadata = {
                        "source_file": base_filename,
                        "normalized_filter_filename": normalized_filter_filename,
                        "full_path": file_path,
                        "page_number": page_number,
                        "conference_name": conference_name,
                        "chunk_type": "text",
                        "chunk_sequence_on_page": i + 1,
                        "source_parser": source_type
                    }
                    all_chunks.append({
                        "content": chunk_text,
                        "metadata": chunk_metadata
                    })
            
            # Treat each table as a separate chunk with a descriptive prefix
            # This assumes `tables_markdown_list` contains the full table MD,
            # potentially including its title if `unstructured` captured it.
            for i, table_md_from_parser in enumerate(tables_markdown_list):
                if table_md_from_parser and table_md_from_parser.strip():
                    clean_table_md = table_md_from_parser.strip()
                    
                    # Try to extract a table name/title if it's at the beginning of the table_md
                    # This is a heuristic. A more robust way is to get it from parser's metadata.
                    table_title_heuristic = ""
                    first_few_lines = clean_table_md.split('\n', 5)
                    if first_few_lines:
                        # Look for lines that might be titles (e.g., "Table X.Y - Title")
                        # before the actual markdown table structure starts (e.g., before "| --- |")
                        potential_title_line = first_few_lines[0]
                        if not potential_title_line.strip().startswith("|") and len(potential_title_line) < 150 : # Avoid taking long paragraphs
                            table_title_heuristic = potential_title_line.strip()

                    descriptive_prefix = (
                        f"The following is a data table extracted from document '{base_filename}', page {page_number}. "
                    )
                    if table_title_heuristic:
                         descriptive_prefix += f"The table is titled or identified as: '{table_title_heuristic}'. "
                    descriptive_prefix += (
                        f"This table presents structured information. The table content is:\n"
                    )
                    
                    table_chunk_content = descriptive_prefix + clean_table_md
                    
                    chunk_metadata = {
                        "source_file": base_filename,
                        "normalized_filter_filename": normalized_filter_filename,
                        "full_path": file_path,
                        "page_number": page_number,
                        "conference_name": conference_name,
                        "chunk_type": "table_markdown_with_description",
                        "table_title_heuristic": table_title_heuristic, # Store the heuristic title
                        "chunk_sequence_on_page": i + 1, 
                        "source_parser": source_type
                    }
                    
                    if len(table_chunk_content) > self.chunk_size * 1.5: 
                        logger.warning(
                            f"Table chunk from {base_filename} page {page_number} (Title: {table_title_heuristic}) is quite large "
                            f"({len(table_chunk_content)} chars). Target chunk size: {self.chunk_size}. "
                        )

                    all_chunks.append({
                        "content": table_chunk_content,
                        "metadata": chunk_metadata
                    })
        
        logger.info(f"Created {len(all_chunks)} chunks for document {file_path} (Normalized filter name: '{normalized_filter_filename}')")
        return all_chunks