"""
FastAPI Application Entry Point
Main application setup and configuration
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .api.routes import router as api_router
from .core.config import get_settings
from .core.exceptions import (
    DocumentProcessingError, QueryParsingError,
    RetrievalError, DecisionEngineError, LlamaModelError
)
from .utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting up AI Document Retrieval System...")

    # Create necessary directories
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "documents"), exist_ok=True)
    os.makedirs(os.path.join(settings.DATA_DIR, "vector_store"), exist_ok=True)
    os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down AI Document Retrieval System...")
    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="AI Document Information Retrieval System",
    description="""
    An intelligent document analysis system that uses Large Language Models (LLMs)
    to process natural language queries and retrieve relevant information from
    insurance policy documents.

    ## Features

    * **Document Upload & Processing**: Support for PDF, DOCX, TXT, and HTML files
    * **Natural Language Queries**: Ask questions in plain English about your documents
    * **Semantic Search**: Advanced vector-based document retrieval using ChromaDB
    * **AI Decision Making**: Automated insurance claim decisions with justifications
    * **LLM Integration**: Meta-Llama-3-8B model for advanced reasoning and explanations
    * **RESTful API**: Full REST API with comprehensive endpoints

    ## Use Cases

    * Insurance claim processing and approval
    * Policy document analysis and interpretation
    * Coverage verification and eligibility checks
    * Automated decision making with audit trails
    """,
    version="1.0.0",
    contact={
        "name": "AI Document Retrieval Team",
        "email": "support@aidocretrieval.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(
    api_router,
    prefix="/api",
    tags=["Document Retrieval API"]
)

# Static files (for serving uploaded documents if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global exception handlers
@app.exception_handler(DocumentProcessingError)
async def document_processing_exception_handler(request: Request, exc: DocumentProcessingError):
    """Handle document processing errors"""
    logger.error(f"Document processing error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Document Processing Error",
            "message": str(exc),
            "type": "document_processing_error"
        }
    )

@app.exception_handler(QueryParsingError)
async def query_parsing_exception_handler(request: Request, exc: QueryParsingError):
    """Handle query parsing errors"""
    logger.error(f"Query parsing error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Query Parsing Error",
            "message": str(exc),
            "type": "query_parsing_error"
        }
    )

@app.exception_handler(RetrievalError)
async def retrieval_exception_handler(request: Request, exc: RetrievalError):
    """Handle retrieval errors"""
    logger.error(f"Retrieval error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Document Retrieval Error",
            "message": str(exc),
            "type": "retrieval_error"
        }
    )

@app.exception_handler(DecisionEngineError)
async def decision_engine_exception_handler(request: Request, exc: DecisionEngineError):
    """Handle decision engine errors"""
    logger.error(f"Decision engine error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Decision Engine Error",
            "message": str(exc),
            "type": "decision_engine_error"
        }
    )

@app.exception_handler(LlamaModelError)
async def llama_model_exception_handler(request: Request, exc: LlamaModelError):
    """Handle Llama model errors"""
    logger.error(f"Llama model error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "AI Model Error",
            "message": str(exc),
            "type": "llama_model_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "type": "internal_error"
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "AI Document Information Retrieval System",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "api_base": "/api/v1",
        "endpoints": {
            "health": "/api/v1/health",
            "upload": "/api/v1/upload",
            "query": "/api/v1/query",
            "documents": "/api/v1/documents",
            "statistics": "/api/v1/statistics"
        },
        "features": [
            "Document upload and processing",
            "Natural language query processing",
            "Semantic document retrieval",
            "AI-powered decision making",
            "LLM integration for advanced reasoning"
        ]
    }

# Health check endpoint (also available at root level)
@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy", "message": "System operational"}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests"""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} in {process_time:.4f}s")

    return response

# Add request ID middleware for tracing
import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response

app.add_middleware(RequestIDMiddleware)

# Development server startup
def start_server():
    """Start the development server"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
