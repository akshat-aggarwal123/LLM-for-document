from time import time
import time
from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from collections import Counter
from pydantic import BaseModel

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)

class SemanticRetriever:
    """Semantic search and retrieval using ChromaDB and sentence transformers."""
    def store_document(self, content, metadata):
        """
        Store document content and metadata in the vector database.
        Returns a unique document ID.
        """
        try:
            # Create a unique document ID
            doc_id = f"doc_{int(time.time())}"
            
            # Process metadata to ensure only simple types are included
            processed_metadata = {}
            
            # Add timestamp to metadata in ISO format
            processed_metadata['upload_date'] = datetime.now().isoformat()
            
            # Process other metadata fields
            if isinstance(metadata, dict):
                # Extract filename if available
                if 'file_path' in metadata:
                    processed_metadata['filename'] = Path(metadata['file_path']).name
                    processed_metadata['file_path'] = metadata['file_path']
                
                # Handle other metadata fields
                for key, value in metadata.items():
                    if key not in processed_metadata:  # Don't overwrite already processed fields
                        # Only include simple types
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            processed_metadata[key] = value
                        else:
                            # Convert complex types to string representation if needed
                            processed_metadata[key] = str(value)
                
                # Ensure we have document type
                if 'document_type' not in processed_metadata:
                    if 'file_path' in processed_metadata:
                        ext = Path(processed_metadata['file_path']).suffix.lower()
                        processed_metadata['document_type'] = ext[1:] if ext else 'unknown'
                    else:
                        processed_metadata['document_type'] = 'unknown'
            
            # Store the document
            self.collection.add(
                documents=[content],
                metadatas=[processed_metadata],
                ids=[doc_id]
            )
            
            logger.info(f"Successfully stored document with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise
    
    def __init__(self, settings):
        self.settings = settings
        self.chroma_client = None
        self.collection = None
        self._initialize_components()

    def _initialize_components(self):
        """Initializes the ChromaDB client and gets or creates the collection."""
        try:
            logger.info(f"Initializing ChromaDB client with persist directory: {self.settings.chroma_persist_dir}")
            vector_store_path = Path(self.settings.chroma_persist_dir)
            vector_store_path.mkdir(exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(path=str(vector_store_path))

            # Use the official ChromaDB utility for embedding functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.embedding_model_name
            )

            # Use the robust 'get_or_create_collection' method
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.settings.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}  # Good for sentence embeddings
            )

            logger.info(f"Successfully connected to collection: '{self.settings.collection_name}'")

        except Exception as e:
            logger.error(f"Failed to initialize semantic retriever: {e}", exc_info=True)
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Adds a batch of document chunks to the vector store."""
        if not documents:
            logger.warning("No documents provided for indexing.")
            return False

        try:
            texts, metadatas, ids = [], [], []
            for doc in documents:
                # Use a stable hashing algorithm for deterministic IDs
                doc_id = f"doc_{hashlib.sha256(doc['content'].encode('utf-8')).hexdigest()}"
                ids.append(doc_id)
                texts.append(doc['content'])
                metadatas.append({k: v for k, v in doc.items() if k != 'content'})

            self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
            logger.info(f"Successfully added {len(documents)} documents to vector store.")
            return True

        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=True)
            return False

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Performs semantic search for a given query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if not results['ids'][0]:
                return []

            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                formatted_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': max(0, 1 - distance), # Cosine distance to similarity
                    'rank': i + 1
                })

            logger.info(f"Retrieved {len(formatted_results)} documents for query.")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Gets statistics about the document collection."""
        try:
            count = self.collection.count()
            if count == 0:
                return {'total_documents': 0}

            # Use collections.Counter for more efficient counting
            all_metadata = self.collection.get(include=["metadatas"])['metadatas']
            doc_types = Counter(meta.get('document_type', 'unknown') for meta in all_metadata)
            sections = Counter(meta.get('section', 'general') for meta in all_metadata)

            return {
                'total_documents': count,
                'collection_name': self.settings.collection_name,
                'embedding_model': self.settings.embedding_model_name,
                'document_types_dist': dict(doc_types),
                'sections_dist': dict(sections)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}", exc_info=True)
            return {'error': str(e)}

    # ... Other methods like clear_collection, search_by_claim_type, etc. can remain ...
    # They will benefit from the more robust core methods established above.

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete collection
            self.chroma_client.delete_collection(self.collection_name)

            # Recreate empty collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(f"Cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

    def update_document(self, doc_id: str, new_content: str, new_metadata: Dict[str, Any]) -> bool:
        """Update an existing document"""
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[new_content],
                metadatas=[new_metadata]
            )

            logger.info(f"Updated document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """Delete specific documents"""
        try:
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def hybrid_search(self, query: str, keywords: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Combine semantic search with keyword filtering

        Args:
            query: Semantic search query
            keywords: Must-have keywords
            top_k: Number of results

        Returns:
            Filtered and ranked results
        """
        try:
            # Perform semantic search with larger result set
            semantic_results = self.search(query, top_k * 2)

            # Filter by keywords
            filtered_results = []
            for result in semantic_results:
                content_lower = result['content'].lower()

                # Check if all keywords are present
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
                keyword_score = keyword_matches / len(keywords) if keywords else 1.0

                if keyword_score > 0:  # At least one keyword match
                    result['keyword_score'] = keyword_score
                    result['hybrid_score'] = (result['similarity_score'] + keyword_score) / 2
                    filtered_results.append(result)

            # Sort by hybrid score and return top_k
            filtered_results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            return filtered_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    def list_stored_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents with their metadata"""
        try:
            # Get all documents with their metadata
            result = self.collection.get(
                include=["documents", "metadatas"]
            )
            
            documents = []
            
            # Process documents and metadata
            for idx, (content, metadata) in enumerate(zip(
                result['documents'], 
                result['metadatas']
            )):
                # Ensure upload_date is a valid ISO format
                upload_date = metadata.get('upload_date')
                if not upload_date or not isinstance(upload_date, str):
                    # Default to current time in ISO format
                    upload_date = datetime.now().isoformat()
                
                # Get filename from metadata or generate one
                filename = metadata.get('filename')
                if not filename:
                    # Try to get filename from original file path if it exists
                    file_path = metadata.get('file_path')
                    if file_path:
                        filename = Path(file_path).name
                    else:
                        filename = f"document_{idx}.txt"
                
                doc_info = {
                    'document_id': metadata.get('file_path', f'doc_{idx}'),  # Use file_path as ID or generate one
                    'filename': filename,
                    'document_type': metadata.get('document_type', 'pdf'),  # Default to PDF since we're handling PDF files
                    'upload_date': upload_date,
                    'sections_count': len(metadata.get('sections', [])),
                    'file_size': metadata.get('file_size', len(content) if content else 0)  # Use content length if file_size not provided
                }
                documents.append(doc_info)
            
            logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents

        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def retrieve_relevant_content(self, query, parsed_data=None, top_k=5):
        """
        Retrieve relevant documents/chunks for a query.
        Returns a dict with 'documents' and optionally 'processing_time'.
        """
        start_time = time.time()
        # Use your semantic search logic here
        results = self.search(query, top_k=top_k)
        processing_time = time.time() - start_time
        return {
            "documents": results,
            "processing_time": processing_time
        }

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
