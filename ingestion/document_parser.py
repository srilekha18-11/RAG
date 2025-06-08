import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import warnings

from pypdf import PdfReader as PyPdfReader # Renamed to avoid conflict
from unstructured.partition.pdf import partition_pdf
# from unstructured.documents.elements import Table # Not directly needed if we use el.category
from unstructured.cleaners.core import clean_extra_whitespace
from google.cloud import vision # Keep for vision_client check, but unstructured handles OCR call
import pandas as pd

# from config import FORCE_VISION_FOR_TABLES # We'll simplify and remove this for now

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    def __init__(self):
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                # Test if client can be initialized (valid credentials)
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Google Vision Client initialized (credentials seem valid). `unstructured` may use it for OCR.")
            except Exception as e:
                self.vision_client = None
                logger.warning(f"Google Vision Client could not be initialized: {e}. `unstructured` will not use it for OCR.")
        else:
            self.vision_client = None
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. `unstructured` will not use Google Vision for OCR. OCR will rely on Tesseract if available.")

    def _is_pdf_scanned(self, file_path: str, threshold_chars: int = 100) -> bool:
        """Rudimentary check if a PDF is scanned by trying to extract text."""
        try:
            reader = PyPdfReader(file_path)
            text_content = ""
            if not reader.pages: # Handle empty or unreadable PDFs
                return True # Assume scanned or problematic
                
            for page_num, page in enumerate(reader.pages):
                # Limit text extraction to first few pages for speed
                if page_num >= 3 and len(text_content) > threshold_chars: # Check first 3 pages
                    break
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text
                if len(text_content) > threshold_chars:
                    logger.debug(f"PDF {file_path} considered not scanned based on text quantity.")
                    return False
            
            is_scanned_result = len(text_content) < threshold_chars
            logger.debug(f"PDF {file_path} scanned check: text_length={len(text_content)}, is_scanned={is_scanned_result}")
            return is_scanned_result
        except Exception as e:
            logger.warning(f"Could not read PDF {file_path} for scanned check with PyPdfReader: {e}. Assuming scanned.")
            return True

    def _parse_pdf_elements_unstructured(self, file_path: str, is_scanned: bool) -> List[Dict[str, any]]:
        """
        Parses a PDF using unstructured library, extracts text and tables.
        Uses "hi_res" strategy. If GOOGLE_APPLICATION_CREDENTIALS are set,
        unstructured may use Google Vision for OCR parts.
        """
        results = []
        elements = []
        try:
            # "hi_res" strategy attempts to use layout models for text PDFs
            # and OCR (like Vision if configured, or Tesseract) for scanned/image parts.
            # It's generally the best for table extraction.
            # If GOOGLE_APPLICATION_CREDENTIALS are set and unstructured[gcp] is installed,
            # unstructured's hi_res strategy can leverage it for OCR.
            logger.info(f"Partitioning PDF {file_path} with unstructured strategy='hi_res' (scanned_hint={is_scanned})")
            elements = partition_pdf(
                filename=file_path,
                strategy="hi_res",
                infer_table_structure=True,
                # ocr_languages="eng", # Relevant if Tesseract is primary OCR
                # pdf_infer_table_structure=True, # Redundant with infer_table_structure
                # For scanned PDFs, unstructured with hi_res should automatically use OCR.
                # If vision_client is available (credentials set), it *should* prefer it.
            )
            
        except Exception as e:
            logger.error(f"Error partitioning PDF {file_path} with unstructured: {e}")
            logger.info("Attempting fallback strategy='fast' for {file_path}")
            try:
                elements = partition_pdf(
                    filename=file_path,
                    strategy="fast", # Fallback, less accurate for tables
                    infer_table_structure=True,
                )
            except Exception as e2:
                logger.error(f"Fallback strategy='fast' also failed for PDF {file_path}: {e2}")
                return [] # Return empty if both fail

        page_data = {} # page_number -> {"text_content": str, "tables_markdown": List[str]}

        for el in elements:
            page_num = el.metadata.page_number if hasattr(el.metadata, 'page_number') and el.metadata.page_number is not None else 1
            
            if page_num not in page_data:
                page_data[page_num] = {"text_content": "", "tables_markdown": []}

            if el.category == "Table":
                table_html = getattr(el.metadata, 'text_as_html', None)
                if table_html:
                    try:
                        # Suppress warnings from read_html if any (e.g. no table found)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            # Ensure html is wrapped in <table> tags for pandas
                            if not table_html.strip().lower().startswith("<table"):
                                html_for_pandas = f"<table>{table_html}</table>"
                            else:
                                html_for_pandas = table_html
                                
                            df_list = pd.read_html(html_for_pandas)
                        
                        if df_list:
                            # Concatenate if multiple tables are parsed from one HTML block, though unlikely
                            table_md = "\n\n".join([df.to_markdown(index=False) for df in df_list])
                            page_data[page_num]["tables_markdown"].append(clean_extra_whitespace(table_md))
                            logger.debug(f"Successfully converted table HTML to Markdown on page {page_num} of {file_path}")
                        else:
                            logger.warning(f"pd.read_html returned empty list for table HTML on page {page_num} of {file_path}. Using raw text.")
                            page_data[page_num]["text_content"] += "Table (structure may be approximate):\n" + el.text + "\n\n"
                    except Exception as e_table:
                        logger.warning(f"Could not convert HTML table to MD for {file_path} page {page_num}: {e_table}. Using raw text.")
                        page_data[page_num]["text_content"] += "Table (structure conversion failed):\n" + el.text + "\n\n"
                else: 
                    logger.info(f"Table element found on page {page_num} of {file_path} but no HTML representation. Using raw text.")
                    page_data[page_num]["text_content"] += "Table (no HTML structure):\n" + el.text + "\n\n"
            else:
                page_data[page_num]["text_content"] += el.text + "\n\n"
        
        for page_num, data in sorted(page_data.items()):
            results.append({
                "page_number": page_num,
                "text_content": clean_extra_whitespace(data["text_content"]),
                "tables_markdown": data["tables_markdown"],
                "source_type": "unstructured_hi_res" + ("_ocr_attempted" if is_scanned else "_text")
            })
            
        if not elements:
            logger.warning(f"No elements extracted by unstructured for {file_path}.")
        elif not results:
             logger.warning(f"Elements extracted for {file_path}, but no page data compiled. Check element processing logic.")


        return results

    def parse_document(self, file_path: str) -> List[Dict[str, any]]:
        """
        Parses a single document (PDF).
        Returns a list of page contents, where each item is a dict:
        {
            "page_number": int,
            "text_content": str,
            "tables_markdown": List[str] (list of tables found on page, in Markdown)
            "source_type": str
        }
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        _, extension = os.path.splitext(file_path)
        if extension.lower() != ".pdf":
            logger.warning(f"Unsupported file type: {extension} for file {file_path}")
            return []

        # The `is_scanned` check here is more of a hint for logging/debugging.
        # `unstructured` with `strategy="hi_res"` will make its own determination
        # on whether to use OCR for parts or all of the document.
        is_scanned_hint = self._is_pdf_scanned(file_path)
        logger.info(f"Processing {file_path} (Scanned hint: {is_scanned_hint}, Vision client available: {bool(self.vision_client)})")

        # Always use the unified unstructured parsing method.
        # It will use Vision for OCR if GOOGLE_APPLICATION_CREDENTIALS are set and
        # it deems OCR necessary (e.g., for scanned PDFs or image regions).
        return self._parse_pdf_elements_unstructured(file_path, is_scanned_hint)

# Example Usage (for testing this module)
if __name__ == '__main__':
    parser = DocumentParser()

    # --- Test with your ISO PDF ---
    # Make sure this path is correct
    test_iso_pdf_path = "data/ASME/D4191.pdf" # Path from your log
    
    if os.path.exists(test_iso_pdf_path):
        print(f"\n--- Parsing Specific PDF: {test_iso_pdf_path} ---")
        parsed_content = parser.parse_document(test_iso_pdf_path)
        if not parsed_content:
            print("No content parsed.")
        for i, page_info in enumerate(parsed_content):
            print(f"Page {page_info['page_number']} (Source: {page_info['source_type']}) Parsed Data {i+1}/{len(parsed_content)}")
            print(f"  Text Length: {len(page_info['text_content'])}")
            if page_info['tables_markdown']:
                print(f"  Tables Found ({len(page_info['tables_markdown'])}):")
                for table_idx, table_md in enumerate(page_info['tables_markdown']):
                    print(f"    --- Table {table_idx+1} ---")
                    print(table_md) # Print full table markdown for inspection
                    print(f"    --- End Table {table_idx+1} ---")
            else:
                print("  No tables found on this page.")
            # print(f"  Page Text Snippet: {page_info['text_content'][:300]}...") # Optional: print text snippet
            print("-" * 30)
    else:
        print(f"Test PDF not found: {test_iso_pdf_path}. Please check the path.")

    # Add other test PDF paths if you have them (e.g., a known scanned PDF)
    # test_scanned_pdf = "path/to/your/scanned_document.pdf"
    # if os.path.exists(test_scanned_pdf):
    #     # ... same parsing and printing logic ...
    #     pass