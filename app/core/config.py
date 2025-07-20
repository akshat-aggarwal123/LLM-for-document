from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic_settings import BaseSettings
import os
from functools import lru_cache # <--- ADD THIS IMPORT

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Configuration
    app_name: str = "Document Analysis API"
    app_version: str = "1.0.0"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    # Model Configuration
    llama_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    huggingface_token: str = ""

    # ChromaDB Configuration
    chroma_persist_dir: str = "./data/vector_store"
    collection_name: str = "document_chunks"
    MODEL_CACHE_DIR: str = "./data/model_cache"

    # Document Processing
    max_chunk_size: int = 1000
    chunk_overlap: int = 150
    max_file_size: int = 50 

    # Decision Engine Parameters
    confidence_threshold: float = 0.7
    retrieval_k: int = 5
    temperature: float = 0.2

    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"


    ALLOWED_ORIGINS: List[str] = ["*"]
    DATA_DIR: str = os.path.join(os.getcwd(), "data")

    class Config:
        env_file = ".env"

# --- ALL THE CODE BELOW THIS LINE IS WHAT YOU ARE ADDING ---

@lru_cache()
def get_settings():
    """
    Returns a cached instance of the Settings object.
    The @lru_cache decorator ensures this function is only run once,
    improving performance.
    """
    return Settings()

# Note: The other dataclasses and constants can remain as they are.
# The key is adding the get_settings() function.

@dataclass
class QueryEntity:
    """Structured representation of extracted query information"""
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    procedure: Optional[str] = None
    condition: Optional[str] = None
    policy_duration: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    raw_query: str = ""


@dataclass
class DocumentClause:
    """Represents a document clause with metadata"""
    id: str
    text: str
    clause_type: str  # coverage, exclusion, payment, eligibility, general
    source_document: str
    page_number: Optional[int] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class DecisionResponse:
    """Final decision response structure"""
    decision: str  # approved, rejected, requires_review
    amount: Optional[float] = None
    confidence_score: float = 0.0
    justification: str = ""
    relevant_clauses: List[DocumentClause] = None
    processing_time: float = 0.0
    query_entities: Optional[QueryEntity] = None

    def __post_init__(self):
        if self.relevant_clauses is None:
            self.relevant_clauses = []

# Supported file types
SUPPORTED_FILE_TYPES = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.eml': 'message/rfc822'
}

# Clause type mappings
CLAUSE_TYPES = {
    'coverage': ['coverage', 'benefit', 'include', 'cover', 'entitled'],
    'exclusion': ['exclusion', 'exclude', 'not covered', 'except', 'limitation'],
    'payment': ['payment', 'premium', 'deductible', 'copay', 'amount'],
    'eligibility': ['eligibility', 'qualify', 'eligible', 'requirement', 'condition'],
    'general': ['policy', 'term', 'definition', 'procedure']
}

print(f"max_file_size from env: {os.getenv('MAX_FILE_SIZE')}")

settings = get_settings()
