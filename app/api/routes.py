from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import QueryRequest, DecisionResponse
from app.services.document_processor import DocumentProcessor
from app.services.query_parser import QueryParser
from app.services.semantic_retriever import SemanticRetriever
from app.services.decision_engine import DecisionEngine

router = APIRouter()
doc_proc = DocumentProcessor()
parser = QueryParser()
retriever = SemanticRetriever()
engine = DecisionEngine()

@router.post("/upload", summary="Upload documents")
async def upload_docs(files: list[UploadFile] = File(...)):
    paths = []
    for file in files:
        if not file.filename:
            raise HTTPException(400, "Empty filename")
        dest = f"data/documents/{file.filename}"
        with open(dest, "wb") as f:
            f.write(await file.read())
        paths.append(dest)
    retriever.index_documents([(t, p, n) for t, p, n in doc_proc.batch_process(paths)])
    return {"indexed": len(paths)}

@router.post("/query", response_model=DecisionResponse)
async def analyze(req: QueryRequest):
    entity = parser.extract_entities(req.query)
    clauses = retriever.retrieve_relevant_clauses(req.query, entity, top_k=10)
    return engine.make_decision(req.query, entity, clauses)