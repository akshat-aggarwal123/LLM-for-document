"""Custom exceptions for the document analysis system"""

class DocumentAnalysisError(Exception):
    """Base exception for document analysis errors"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DocumentProcessingError(DocumentAnalysisError):
    """Raised when document processing fails"""
    pass

class LlamaModelError(Exception):
    """Custom exception for Llama model related errors."""
    pass
class RetrievalError(Exception):
    """Custom exception for document retrieval errors."""
    pass

class QueryParsingError(DocumentAnalysisError):
    """Raised when query parsing fails"""
    pass


class ModelLoadError(DocumentAnalysisError):
    """Raised when model loading fails"""
    pass


class VectorStoreError(DocumentAnalysisError):
    """Raised when vector store operations fail"""
    pass


class DecisionEngineError(DocumentAnalysisError):
    """Raised when decision engine fails"""
    pass


class FileTypeNotSupportedError(DocumentAnalysisError):
    """Raised when unsupported file type is uploaded"""
    pass


class FileSizeExceededError(DocumentAnalysisError):
    """Raised when file size exceeds limit"""
    pass
