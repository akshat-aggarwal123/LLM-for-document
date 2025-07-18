from pydantic import BaseModel
from typing import Optional, List

class QueryRequest(BaseModel):
    query: str

class Clause(BaseModel):
    content: str
    relevance_score: float
    source_document: str
    page_number: int

class DecisionResponse(BaseModel):
    decision: str
    amount: Optional[float] = None
    justification: str
    relevant_clauses: List[Clause] = []