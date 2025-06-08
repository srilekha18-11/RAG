import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging

from config import VECTOR_STORE_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        
        # Using GoogleGenerativeAIEmbeddingFunction from chromadb.utils
        # This requires google-generativeai to be installed
        self.embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=GOOGLE_API_KEY,
            model_name=EMBEDDING_MODEL_NAME # e.g., "models/embedding-001" or "models/text-embedding-004"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"} # Good default for semantic similarity
        )
        logger.info(f"ChromaDB client initialized. Collection '{CHROMA_COLLECTION_NAME}' loaded/created.")

    def add_chunks_to_store(self, chunks_with_metadata: List[Dict[str, Any]], batch_size: int = 100):
        """Adds chunks to the vector store. Chunks are dicts with 'content' and 'metadata'."""
        if not chunks_with_metadata:
            logger.info("No chunks to add.")
            return

        contents = [chunk['content'] for chunk in chunks_with_metadata]
        metadatas = [chunk['metadata'] for chunk in chunks_with_metadata]
        ids = [f"{chunk['metadata']['full_path']}_page{chunk['metadata']['page_number']}_type{chunk['metadata']['chunk_type']}_seq{idx}" 
               for idx, chunk in enumerate(chunks_with_metadata)] # Create unique IDs

        num_chunks = len(contents)
        for i in range(0, num_chunks, batch_size):
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                self.collection.add(
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"Added batch of {len(batch_ids)} chunks to ChromaDB. Total processed so far: {i + len(batch_ids)}/{num_chunks}")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                # Optionally, add retry logic or save failed batches
        
        logger.info(f"Finished adding {num_chunks} chunks to ChromaDB.")

    def query_store(self, query_text: str, n_results: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """Queries the vector store."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_metadata, # e.g., {"source_file": "paper1.pdf"}
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results nicely
            formatted_results = []
            if results and results.get('ids', [[]])[0]: # Check if results are not empty
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

# Example Usage
if __name__ == '__main__':
    # This assumes GOOGLE_API_KEY is set in .env
    if not GOOGLE_API_KEY:
        print("WARNING: GOOGLE_API_KEY not set. ChromaDB embedding function will fail.")
        exit()

    vsm = VectorStoreManager()
    
    # Dummy chunks for testing
    # test_chunks = [
    #     {"content": "This is about concrete strength.", "metadata": {"source_file": "paperA.pdf", "page_number": 1, "conference_name": "ConcreteConf", "chunk_type": "text", "full_path": "/path/to/paperA.pdf"}},
    #     {"content": "Table shows material properties.", "metadata": {"source_file": "paperA.pdf", "page_number": 2, "conference_name": "ConcreteConf", "chunk_type": "table_markdown", "full_path": "/path/to/paperA.pdf"}},
    #     {"content": "Soil mechanics discussion.", "metadata": {"source_file": "paperB.pdf", "page_number": 5, "conference_name": "GeoConf", "chunk_type": "text", "full_path": "/path/to/paperB.pdf"}},
    # ]
    # vsm.add_chunks_to_store(test_chunks)
    
    # print("\nQuerying for 'concrete':")
    # results_concrete = vsm.query_store("concrete", n_results=2)
    # for res in results_concrete:
    #     print(f"  ID: {res['id']}, Dist: {res['distance']:.4f}, Content: {res['content'][:50]}...")
    #     print(f"    Meta: {res['metadata']}")

    # print("\nQuerying for 'soil' from 'paperB.pdf':")
    # results_soil_filtered = vsm.query_store("soil", n_results=2, filter_metadata={"source_file": "paperB.pdf"})
    # for res in results_soil_filtered:
    #     print(f"  ID: {res['id']}, Dist: {res['distance']:.4f}, Content: {res['content'][:50]}...")
    #     print(f"    Meta: {res['metadata']}")
    
    print(f"VectorStoreManager initialized. Collection count: {vsm.collection.count()}")