"""
API Routes for Document Information Retrieval System
"""

import os
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from fastapi.responses import JSONResponse
import aiofiles

from ..models.schemas import (
    QueryRequest,
    DecisionResponse as QueryResponse,
    DocumentUploadResponse as UploadResponse,
    HealthResponse,
    DocumentInfo,
    ErrorResponse
)
from ..services.document_processor import DocumentProcessor
from ..services.query_parser import QueryParser
from ..services.semantic_retriever import SemanticRetriever
from ..services.decision_engine import DecisionEngine
from ..services.llama_model import LlamaModelService
from ..core.config import get_settings
from ..core.exceptions import (
    DocumentProcessingError, QueryParsingError,
    RetrievalError, DecisionEngineError
)

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

# Initialize services
document_processor = DocumentProcessor(settings)
query_parser = QueryParser(settings)
semantic_retriever = SemanticRetriever(settings)
decision_engine = DecisionEngine(settings)
llama_model = LlamaModelService(settings)

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check if all services are initialized
        services_status = {
            "document_processor": document_processor is not None,
            "query_parser": query_parser is not None,
            "semantic_retriever": semantic_retriever is not None,
            "decision_engine": decision_engine is not None,
            "llama_model": llama_model.model is not None,
            "vector_db": semantic_retriever.collection is not None
        }

        all_healthy = all(services_status.values())

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            message="All systems operational" if all_healthy else "Some services unavailable",
            services=services_status
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"System error: {str(e)}",
            services={}
        )

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.html'}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        # Create documents directory if it doesn't exist
        documents_dir = os.path.join(settings.DATA_DIR, "documents")
        os.makedirs(documents_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(documents_dir, file.filename)

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"File saved: {file_path}")

        # Process document
        processed_data = document_processor.process_document(file_path)

        # Store in vector database
        document_id = semantic_retriever.store_document(
            content=processed_data['content'],
            metadata={
                'filename': file.filename,
                'file_path': file_path,
                'document_type': processed_data['document_type'],
                'sections': processed_data['sections'],
                'entities': processed_data['entities']
            }
        )

        return UploadResponse(
            filename=file.filename,
            document_id=document_id,
            message="Document uploaded and processed successfully",
            document_type=processed_data['document_type'],
            sections_extracted=len(processed_data['sections']),
            entities_found=len(processed_data['entities'])
        )

    except DocumentProcessingError as e:
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document processing failed: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language queries against uploaded documents"""

    try:
        logger.info(f"Processing query: {request.query}")

        # Parse the query
        parsed_query = query_parser.parse_query(request.query)
        logger.info(f"Parsed query: {parsed_query}")

        # Retrieve relevant documents
        retrieval_results = semantic_retriever.retrieve_relevant_content(
            query=request.query,
            parsed_data=parsed_query,
            top_k=request.top_k or 5
        )
        logger.info(f"Retrieved {len(retrieval_results['documents'])} relevant documents")

        # Make decision using decision engine
        decision_result = decision_engine.make_decision(
            query_data=parsed_query,
            relevant_documents=retrieval_results['documents'],
            user_context=request.context or {}
        )

        # Generate advanced reasoning with Llama model if needed
        if request.use_llm and llama_model.model is not None:
            try:
                llm_reasoning = llama_model.generate_decision_reasoning(
                    query_data=parsed_query,
                    relevant_clauses=retrieval_results['documents'],
                    preliminary_decision=decision_result['decision']
                )

                # Enhanced justification
                enhanced_justification = llama_model.generate_justification(
                    decision=decision_result['decision'],
                    query_data=parsed_query,
                    relevant_clauses=retrieval_results['documents'],
                    amount=decision_result.get('amount')
                )

                # Merge LLM insights with decision result
                decision_result.update({
                    'detailed_reasoning': llm_reasoning.get('detailed_reasoning', ''),
                    'confidence_factors': llm_reasoning.get('confidence_factors', []),
                    'risk_assessment': llm_reasoning.get('risk_assessment', 'medium'),
                    'justification': enhanced_justification
                })

            except Exception as llm_error:
                logger.warning(f"LLM enhancement failed: {str(llm_error)}")
                # Continue with standard decision result

        # Prepare response
        response = QueryResponse(
            query=request.query,
            decision=decision_result['decision'],
            confidence_score=decision_result['confidence_score'],
            justification=decision_result['justification'],
            amount=decision_result.get('amount'),
            relevant_clauses=decision_result['relevant_clauses'],
            parsed_entities=parsed_query,
            processing_time=retrieval_results.get('processing_time', 0.0),
            documents_searched=len(retrieval_results['documents']),
            detailed_reasoning=decision_result.get('detailed_reasoning'),
            confidence_factors=decision_result.get('confidence_factors', []),
            risk_assessment=decision_result.get('risk_assessment')
        )

        logger.info(f"Query processed successfully. Decision: {response.decision}")
        return response

    except QueryParsingError as e:
        logger.error(f"Query parsing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query parsing failed: {str(e)}"
        )

    except RetrievalError as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document retrieval failed: {str(e)}"
        )

    except DecisionEngineError as e:
        logger.error(f"Decision engine error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Decision processing failed: {str(e)}"
        )

    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""

    try:
        documents = semantic_retriever.list_stored_documents()

        document_info = []
        for doc in documents:
            metadata = doc.get('metadata', {})
            document_info.append(DocumentInfo(
                document_id=doc.get('id', ''),
                filename=metadata.get('filename', 'Unknown'),
                document_type=metadata.get('document_type', 'unknown'),
                upload_date=metadata.get('upload_date', ''),
                sections_count=len(metadata.get('sections', [])),
                file_size=metadata.get('file_size', 0)
            ))

        return document_info

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document"""

    try:
        success = semantic_retriever.delete_document(document_id)

        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/analyze-document")
async def analyze_document_content(file: UploadFile = File(...)):
    """Analyze document content without storing it"""

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    try:
        # Read file content
        content = await file.read()

        # Create temporary file
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)

        # Process document
        processed_data = document_processor.process_document(temp_path)

        # Analyze with Llama model
        analysis_result = {}
        if llama_model.model is not None:
            analysis_result = llama_model.analyze_document_content(
                content=processed_data['content'],
                document_type=processed_data['document_type']
            )

        # Clean up temp file
        os.remove(temp_path)

        return {
            "filename": file.filename,
            "document_type": processed_data['document_type'],
            "content_summary": processed_data['content'][:500] + "...",
            "sections": processed_data['sections'],
            "entities": processed_data['entities'],
            "analysis": analysis_result
        }

    except Exception as e:
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}"
        )

@router.post("/batch-query")
async def process_batch_queries(queries: List[str], use_llm: bool = False):
    """Process multiple queries in batch"""

    if len(queries) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 queries allowed per batch"
        )

    results = []

    for i, query in enumerate(queries):
        try:
            request = QueryRequest(query=query, use_llm=use_llm)
            result = await process_query(request)
            results.append({
                "query_id": i,
                "query": query,
                "result": result
            })

        except Exception as e:
            results.append({
                "query_id": i,
                "query": query,
                "error": str(e)
            })

    return {
        "batch_size": len(queries),
        "processed": len([r for r in results if "result" in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results
    }

@router.get("/statistics")
async def get_system_statistics():
    """Get system usage statistics"""

    try:
        stats = {
            "total_documents": semantic_retriever.get_document_count(),
            "total_queries_processed": 0,  # Would need to track this in a database
            "vector_db_size": semantic_retriever.get_collection_size(),
            "average_response_time": 0.0,  # Would need to track this
            "model_status": {
                "llama_loaded": llama_model.model is not None,
                "device": llama_model.device,
                "model_name": "Meta-Llama-3-8B-Instruct"
            }
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
