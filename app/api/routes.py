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
from ..services.decision_engine import DecisionEngine, DecisionType
from ..services.llama_model import LlamaModelService
from ..core.config import get_settings
from ..core.exceptions import (
    DocumentProcessingError, QueryParsingError,
    RetrievalError, DecisionEngineError
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Get settings
print("[DEBUG] Getting settings...")
settings = get_settings()
print(f"[DEBUG] Settings obtained: {type(settings)}")

# Initialize services with proper error handling
print("[DEBUG] === SERVICE INITIALIZATION STARTED ===")

# Initialize DocumentProcessor
try:
    print("[DEBUG] Initializing DocumentProcessor...")
    document_processor = DocumentProcessor(settings)
    print("[DEBUG] DocumentProcessor initialized successfully")
except Exception as e:
    print(f"[DEBUG] DocumentProcessor initialization failed: {str(e)}")
    logger.error(f"DocumentProcessor initialization failed: {str(e)}")
    document_processor = None

# Initialize QueryParser
try:
    print("[DEBUG] Initializing QueryParser...")
    query_parser = QueryParser(settings)
    print("[DEBUG] QueryParser initialized successfully")
except Exception as e:
    print(f"[DEBUG] QueryParser initialization failed: {str(e)}")
    logger.error(f"QueryParser initialization failed: {str(e)}")
    query_parser = None

# Initialize SemanticRetriever
try:
    print("[DEBUG] Initializing SemanticRetriever...")
    semantic_retriever = SemanticRetriever(settings)
    print("[DEBUG] SemanticRetriever initialized successfully")
except Exception as e:
    print(f"[DEBUG] SemanticRetriever initialization failed: {str(e)}")
    logger.error(f"SemanticRetriever initialization failed: {str(e)}")
    semantic_retriever = None

# Initialize DecisionEngine
try:
    print("[DEBUG] Initializing DecisionEngine...")
    decision_engine = DecisionEngine(settings)
    print("[DEBUG] DecisionEngine initialized successfully")
except Exception as e:
    print(f"[DEBUG] DecisionEngine initialization failed: {str(e)}")
    logger.error(f"DecisionEngine initialization failed: {str(e)}")
    decision_engine = None

# Initialize LlamaModelService
try:
    print("[DEBUG] Initializing LlamaModelService...")
    llama_model = LlamaModelService(settings)
    print("[DEBUG] LlamaModelService initialized successfully")
except Exception as e:
    print(f"[DEBUG] LlamaModelService initialization failed: {str(e)}")
    logger.error(f"LlamaModelService initialization failed: {str(e)}")
    llama_model = None

print("[DEBUG] === SERVICE INITIALIZATION COMPLETED ===")
print(f"[DEBUG] DocumentProcessor available: {document_processor is not None}")
print(f"[DEBUG] QueryParser available: {query_parser is not None}")
print(f"[DEBUG] SemanticRetriever available: {semantic_retriever is not None}")
print(f"[DEBUG] DecisionEngine available: {decision_engine is not None}")
print(f"[DEBUG] LlamaModelService available: {llama_model is not None}")

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    print("[DEBUG] === HEALTH CHECK STARTED ===")
    
    try:
        # Check if all services are initialized
        services_status = {
            "document_processor": document_processor is not None,
            "query_parser": query_parser is not None,
            "semantic_retriever": semantic_retriever is not None,
            "decision_engine": decision_engine is not None,
            "llama_model": llama_model is not None and hasattr(llama_model, 'model') and llama_model.model is not None,
            "vector_db": semantic_retriever is not None and hasattr(semantic_retriever, 'collection') and semantic_retriever.collection is not None
        }

        print(f"[DEBUG] Services status: {services_status}")
        all_healthy = all(services_status.values())

        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            message="All systems operational" if all_healthy else "Some services unavailable",
            services=services_status
        )

    except Exception as e:
        print(f"[DEBUG] Health check error: {str(e)}")
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"System error: {str(e)}",
            services={}
        )



        return HealthResponse(
            status="healthy" if all_healthy else "degraded",
            message="All systems operational" if all_healthy else "Some services unavailable",
            services=services_status
        )

    except Exception as e:
        print(f"[DEBUG] Health check error: {str(e)}")
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            message=f"System error: {str(e)}",
            services={}
        )

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process documents"""
    print("[DEBUG] === DOCUMENT UPLOAD STARTED ===")
    print(f"[DEBUG] Received file: {file.filename}")
    print(f"[DEBUG] File content type: {file.content_type}")

    if not file.filename:
        print("[DEBUG] No filename provided")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.txt', '.html'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    print(f"[DEBUG] File extension: {file_extension}")

    if file_extension not in allowed_extensions:
        print(f"[DEBUG] Invalid file extension: {file_extension}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_extension} not supported. Allowed types: {', '.join(allowed_extensions)}"
        )

    try:
        # Create documents directory if it doesn't exist
        documents_dir = os.path.join(settings.DATA_DIR, "documents")
        print(f"[DEBUG] Documents directory: {documents_dir}")
        os.makedirs(documents_dir, exist_ok=True)
        print("[DEBUG] Documents directory created/verified")

        # Save uploaded file
        file_path = os.path.join(documents_dir, file.filename)
        print(f"[DEBUG] Saving file to: {file_path}")

        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            file_size = len(content)
            print(f"[DEBUG] File size: {file_size} bytes")
            await f.write(content)

        print(f"[DEBUG] File saved successfully: {file_path}")
        logger.info(f"File saved: {file_path}")

        # Check if DocumentProcessor is available
        print("[DEBUG] Starting document processing...")
        if document_processor is None:
            print("[DEBUG] DocumentProcessor not available")
            raise DocumentProcessingError("DocumentProcessor not initialized")

        print("[DEBUG] DocumentProcessor is available, calling process_document...")
        
        # Process document - this now calls your fixed DocumentProcessor
        print("[DEBUG] DocumentProcessor is available, calling process_document...")
        clauses = document_processor.process_document(file_path)

        processed_data = {
            'document_type': 'unknown',  # or infer from clauses
            'sections': clauses,
            'entities': [],  # or extract entities if available
            'content': "\n\n".join([clause.text for clause in clauses])
        }
        print(f"[DEBUG] Document processed successfully")
        print(f"[DEBUG] Processed data keys: {list(processed_data.keys()) if isinstance(processed_data, dict) else 'Not a dict'}")

        # For now, let's create a simple response without semantic_retriever to test DocumentProcessor
        if semantic_retriever is None:
            print("[DEBUG] SemanticRetriever not available, returning basic response")
            return UploadResponse(
                filename=file.filename,
                document_id="temp_id_no_vector_store",
                message="Document processed successfully (vector storage unavailable)",
                document_type=processed_data.get('document_type', 'unknown'),
                sections_extracted=len(processed_data.get('sections', [])),
                entities_found=len(processed_data.get('entities', []))
            )

        # Store in vector database
        print("[DEBUG] Storing document in vector database...")
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
        print(f"[DEBUG] Document stored with ID: {document_id}")

        return UploadResponse(
            filename=file.filename,
            document_id=document_id,
            message="Document uploaded and processed successfully",
            document_type=processed_data['document_type'],
            sections_extracted=len(processed_data['sections']),
            entities_found=len(processed_data['entities']),
            success=True,
            pages_processed=len(processed_data['sections']),
            chunks_created=len(processed_data['sections']),
            processing_time=0.0  # You can set this to the actual processing time if available
        )

    except DocumentProcessingError as e:
        print(f"[DEBUG] Document processing error: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document processing failed: {str(e)}"
        )

    except Exception as e:
        print(f"[DEBUG] Upload error: {str(e)}")
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language queries against uploaded documents"""
    print("[DEBUG] === QUERY PROCESSING STARTED ===")

    # Check required services
    if query_parser is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QueryParser service not available"
        )
    
    if semantic_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRetriever service not available"
        )
        
    if decision_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DecisionEngine service not available"
        )

    try:
        logger.info(f"Processing query: {request.query}")
        print(f"[DEBUG] Query: {request.query}")

        # Parse the query
        parsed_query = query_parser.parse_query(request.query)
        logger.info(f"Parsed query: {parsed_query}")
        print(f"[DEBUG] Parsed query: {parsed_query}")

        # Retrieve relevant documents
        retrieval_results = semantic_retriever.retrieve_relevant_content(
            query=request.query,
            parsed_data=parsed_query,
            top_k=request.top_k or 5
        )
        logger.info(f"Retrieved {len(retrieval_results['documents'])} relevant documents")
        print(f"[DEBUG] Retrieved {len(retrieval_results['documents'])} documents")

        print(f"[DEBUG] claim_info: {parsed_query}")
        print(f"[DEBUG] retrieved_documents: {retrieval_results['documents']}")

        # Make decision using decision engine
        decision_result = decision_engine.make_decision(
            claim_info=parsed_query,
            retrieved_documents=retrieval_results['documents']
        )
        
        print(f"[DEBUG] Decision made: {decision_result.decision}")

        # Generate advanced reasoning with Llama model if needed
        if request.use_llm and llama_model is not None and hasattr(llama_model, 'model') and llama_model.model is not None:
            print("[DEBUG] Using LLM for enhanced reasoning...")
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
                print("[DEBUG] LLM enhancement completed")

            except Exception as llm_error:
                print(f"[DEBUG] LLM enhancement failed: {str(llm_error)}")
                logger.warning(f"LLM enhancement failed: {str(llm_error)}")
                # Continue with standard decision result
        else:
            print("[DEBUG] LLM not available or not requested")

        # Convert relevant_clauses to proper DocumentClauseResponse objects
        formatted_clauses = []
        for idx, clause in enumerate(decision_result.relevant_clauses):
            if isinstance(clause, str):
                # If it's just a string, create a fully compliant clause object
                formatted_clauses.append({
                    "id": f"clause_{idx}",
                    "text": clause,
                    "source_document": "health_insurance_coverage.pdf",  # This should be dynamic based on actual source
                    "confidence": 0.8,
                    "relevance_score": 0.8,
                    "metadata": {
                        "source": "document",
                        "page": 1
                    }
                })
            else:
                # If it's already a dict/object, ensure it has all required fields
                if not isinstance(clause, dict):
                    clause = clause.__dict__  # Convert to dict if it's an object
                # Add any missing required fields
                clause.update({
                    "id": clause.get("id", f"clause_{idx}"),
                    "source_document": clause.get("source_document", "health_insurance_coverage.pdf"),
                    "confidence": clause.get("confidence", 0.8)
                })
                formatted_clauses.append(clause)

        # Format the decision type correctly
        decision = decision_result.decision.value if isinstance(decision_result.decision, DecisionType) else decision_result.decision

        # Prepare response
        response = QueryResponse(
            query=request.query,
            decision=decision,
            confidence_score=decision_result.confidence_score,
            justification=decision_result.justification,
            amount=getattr(decision_result, "approved_amount", None),
            relevant_clauses=formatted_clauses,
            parsed_entities=parsed_query,
            processing_time=retrieval_results.get('processing_time', 0.0),
            documents_searched=len(retrieval_results['documents']),
            detailed_reasoning=getattr(decision_result, "detailed_reasoning", None),
            confidence_factors=getattr(decision_result, "confidence_factors", []),
            risk_assessment=getattr(decision_result, "risk_assessment", None)
        )

        logger.info(f"Query processed successfully. Decision: {response.decision}")
        print(f"[DEBUG] Query processing completed successfully")
        return response

    except QueryParsingError as e:
        print(f"[DEBUG] Query parsing error: {str(e)}")
        logger.error(f"Query parsing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Query parsing failed: {str(e)}"
        )

    except RetrievalError as e:
        print(f"[DEBUG] Retrieval error: {str(e)}")
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document retrieval failed: {str(e)}"
        )

    except DecisionEngineError as e:
        print(f"[DEBUG] Decision engine error: {str(e)}")
        logger.error(f"Error in decision making: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Decision processing failed: {str(e)}"
        )

    except Exception as e:
        print(f"[DEBUG] Query processing error: {str(e)}")
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all uploaded documents"""
    print("[DEBUG] === LISTING DOCUMENTS ===")

    if semantic_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRetriever service not available"
        )

    try:
        documents = semantic_retriever.list_stored_documents()
        print(f"[DEBUG] Found {len(documents)} documents")

        # Documents are already in the correct format from semantic_retriever
        document_info = [DocumentInfo(**doc) for doc in documents]

        return document_info

    except Exception as e:
        print(f"[DEBUG] Error listing documents: {str(e)}")
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a specific document"""
    print(f"[DEBUG] === DELETING DOCUMENT {document_id} ===")

    if semantic_retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SemanticRetriever service not available"
        )

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
        print(f"[DEBUG] Error deleting document: {str(e)}")
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/analyze-document")
async def analyze_document_content(file: UploadFile = File(...)):
    """Analyze document content without storing it"""
    print("[DEBUG] === DOCUMENT ANALYSIS STARTED ===")

    if document_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DocumentProcessor service not available"
        )

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )

    try:
        # Read file content
        content = await file.read()
        print(f"[DEBUG] File content read: {len(content)} bytes")

        # Create temporary file
        temp_path = f"/tmp/{file.filename}"
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)

        # Process document
        processed_data = document_processor.process_document(temp_path)

        # Analyze with Llama model
        analysis_result = {}
        if llama_model is not None and hasattr(llama_model, 'model') and llama_model.model is not None:
            print("[DEBUG] Using LLM for document analysis...")
            analysis_result = llama_model.analyze_document_content(
                content=processed_data['content'],
                document_type=processed_data['document_type']
            )

        # Clean up temp file
        os.remove(temp_path)
        print("[DEBUG] Temporary file cleaned up")

        return {
            "filename": file.filename,
            "document_type": processed_data['document_type'],
            "content_summary": processed_data['content'][:500] + "...",
            "sections": processed_data['sections'],
            "entities": processed_data['entities'],
            "analysis": analysis_result
        }

    except Exception as e:
        print(f"[DEBUG] Document analysis error: {str(e)}")
        logger.error(f"Document analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}"
        )

@router.post("/batch-query")
async def process_batch_queries(queries: List[str], use_llm: bool = False):
    """Process multiple queries in batch"""
    print("[DEBUG] === BATCH QUERY PROCESSING STARTED ===")

    if len(queries) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 queries allowed per batch"
        )

    results = []

    for i, query in enumerate(queries):
        print(f"[DEBUG] Processing batch query {i+1}/{len(queries)}")
        try:
            request = QueryRequest(query=query, use_llm=use_llm)
            result = await process_query(request)
            results.append({
                "query_id": i,
                "query": query,
                "result": result
            })

        except Exception as e:
            print(f"[DEBUG] Batch query {i+1} failed: {str(e)}")
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
    print("[DEBUG] === GETTING SYSTEM STATISTICS ===")

    try:
        stats = {
            "total_documents": semantic_retriever.get_document_count() if semantic_retriever else 0,
            "total_queries_processed": 0,  # Would need to track this in a database
            "vector_db_size": semantic_retriever.get_collection_size() if semantic_retriever else 0,
            "average_response_time": 0.0,  # Would need to track this
            "model_status": {
                "llama_loaded": llama_model is not None and hasattr(llama_model, 'model') and llama_model.model is not None,
                "device": getattr(llama_model, 'device', 'unknown') if llama_model else 'unknown',
                "model_name": "Meta-Llama-3-8B-Instruct"
            },
            "services_available": {
                "document_processor": document_processor is not None,
                "query_parser": query_parser is not None,
                "semantic_retriever": semantic_retriever is not None,
                "decision_engine": decision_engine is not None,
                "llama_model": llama_model is not None
            }
        }

        return stats

    except Exception as e:
        print(f"[DEBUG] Error getting statistics: {str(e)}")
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )
