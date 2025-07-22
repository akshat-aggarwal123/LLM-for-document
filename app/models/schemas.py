"""Pydantic schemas for request/response validation"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Request schema for document queries"""
    query: str = Field(..., min_length=1, max_length=500, description="Natural language query")
    include_clauses: bool = Field(default=True, description="Include relevant clauses in response")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of results to return")
    top_k: int = Field(default=5, ge=1, le=20, description="Top K results to return")
    context: dict = Field(default_factory=dict, description="Contextual information for the query")
    use_llm: bool = Field(default=False, description="Flag to use large language model for query processing")

    @validator('query')
    def validate_query(cls, v):
        if not v or v.isspace():
            raise ValueError('Query cannot be empty or whitespace')
        return v.strip()


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload"""
    success: bool
    message: str
    document_id: str
    filename: str
    pages_processed: int
    chunks_created: int
    processing_time: float


class QueryEntityResponse(BaseModel):
    """Response schema for extracted query entities"""
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    procedure: Optional[str] = None
    condition: Optional[str] = None
    policy_duration: Optional[str] = None
    amount: Optional[float] = None
    date: Optional[str] = None
    raw_query: str


class DocumentClauseResponse(BaseModel):
    """Response schema for document clauses"""
    id: str
    text: str
    clause_type: str
    source_document: str
    page_number: Optional[int] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class DecisionResponse(BaseModel):
    """Response schema for decision results"""
    decision: str = Field(..., description="Decision result: approved, rejected, or requires_review")
    amount: Optional[float] = Field(None, ge=0, description="Approved amount if applicable")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in the decision")
    justification: str = Field(..., description="Detailed justification for the decision")
    relevant_clauses: List[DocumentClauseResponse] = []
    processing_time: float = Field(..., ge=0, description="Time taken to process the query")
    query_entities: Optional[QueryEntityResponse] = None

    @validator('decision')
    def validate_decision(cls, v):
        valid_decisions = ['approved', 'rejected', 'requires_review']
        if v not in valid_decisions:
            raise ValueError(f'Decision must be one of: {valid_decisions}')
        return v


class SystemStatsResponse(BaseModel):
    """Response schema for system statistics"""
    total_documents: int
    total_chunks: int
    supported_file_types: List[str]
    model_info: Dict[str, str]
    uptime: str
    last_updated: datetime


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    error: bool = True
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str = "healthy"
    timestamp: datetime
    version: str
    services: Dict[str, bool]


class DocumentInfo(BaseModel):
    """Response schema for a single document's metadata"""
    id: str = Field(alias='document_id')
    filename: str
    document_type: str
    upload_date: datetime
    sections_count: int
    file_size: int

    @validator('upload_date', pre=True)
    def parse_upload_date(cls, v):
        if not v:  # Handle empty values
            return datetime.now()
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                return datetime.now()  # Fallback to current time if parsing fails
        return v
