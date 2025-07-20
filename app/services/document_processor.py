"""Document processing service for extracting and chunking text from various file formats"""

import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import time

# Document processing libraries
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import email

from app.core.config import DocumentClause, CLAUSE_TYPES
from app.core.exceptions import DocumentProcessingError, FileTypeNotSupportedError
from app.utils.logging import app_logger as logger


class DocumentProcessor:
    """Handles document loading, text extraction, and chunking"""

    def __init__(self, settings):
        self.settings = settings
        self.supported_extensions = ['.pdf', '.docx', '.txt', '.html', '.eml']

    def process_document(self, file_path: str) -> List[DocumentClause]:
        """
        Process a document and return structured clauses

        Args:
            file_path: Path to the document file

        Returns:
            List of DocumentClause objects
        """
        start_time = time.time()

        try:
            # Validate file
            self._validate_file(file_path)

            # Extract text content
            pages_content = self._extract_text(file_path)

            # Create chunks from content
            clauses = self._create_clauses(pages_content, file_path)

            processing_time = time.time() - start_time
            logger.info(f"Processed {len(clauses)} clauses from {file_path} in {processing_time:.2f}s")

            return clauses

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise DocumentProcessingError(f"Failed to process document: {str(e)}")

    def _validate_file(self, file_path: str) -> None:
        """Validate file existence, type, and size"""

        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise FileTypeNotSupportedError(f"Unsupported file type: {file_ext}")

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > settings.max_file_size:
            raise DocumentProcessingError(f"File size ({file_size_mb:.1f}MB) exceeds limit ({settings.max_file_size}MB)")

    def _extract_text(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extract text content from file

        Returns:
            List of tuples (text_content, page_number)
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_ext == '.docx':
            return self._extract_from_docx(file_path)
        elif file_ext == '.txt':
            return self._extract_from_txt(file_path)
        elif file_ext == '.html':
            return self._extract_from_html(file_path)
        elif file_ext == '.eml':
            return self._extract_from_email(file_path)
        else:
            raise FileTypeNotSupportedError(f"Unsupported file extension: {file_ext}")

    def _extract_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF file"""
        pages_content = []

        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        pages_content.append((text.strip(), page_num))

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from PDF: {str(e)}")

        return pages_content

    def _extract_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text_content = []

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text.strip())

            # Combine all text into a single page (DOCX doesn't have clear page breaks)
            full_text = '\n'.join(text_content)
            return [(full_text, 1)] if full_text else []

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from DOCX: {str(e)}")

    def _extract_from_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                return [(content, 1)] if content else []

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from TXT: {str(e)}")

    def _extract_from_html(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from HTML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                text = soup.get_text()
                return [(text.strip(), 1)] if text.strip() else []

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from HTML: {str(e)}")

    def _extract_from_email(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from email file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file)

            # Extract subject and body
            subject = msg.get('Subject', '')

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body += part.get_payload(decode=True).decode('utf-8')
            else:
                body = msg.get_payload(decode=True).decode('utf-8')

            content = f"Subject: {subject}\n\n{body}".strip()
            return [(content, 1)] if content else []

        except Exception as e:
            raise DocumentProcessingError(f"Failed to extract text from email: {str(e)}")

    def _create_clauses(self, pages_content: List[Tuple[str, int]], file_path: str) -> List[DocumentClause]:
        """
        Create DocumentClause objects from extracted text

        Args:
            pages_content: List of (text, page_number) tuples
            file_path: Source file path

        Returns:
            List of DocumentClause objects
        """
        clauses = []
        clause_id = 0

        for text_content, page_num in pages_content:
            # Split text into chunks
            chunks = self._chunk_text(text_content)

            for chunk in chunks:
                if len(chunk.strip()) < 20:  # Skip very short chunks
                    continue

                clause_id += 1
                clause_type = self._classify_clause(chunk)

                clause = DocumentClause(
                    id=f"{Path(file_path).stem}_{clause_id}",
                    text=chunk,
                    clause_type=clause_type,
                    source_document=os.path.basename(file_path),
                    page_number=page_num,
                    metadata={
                        'chunk_length': len(chunk),
                        'file_path': file_path
                    }
                )
                clauses.append(clause)

        return clauses

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using sentence boundaries

        Args:
            text: Input text to chunk

        Returns:
            List of text chunks
        """
        # Split into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) <= settings.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _classify_clause(self, text: str) -> str:
        """
        Classify clause type based on keywords

        Args:
            text: Clause text

        Returns:
            Clause type string
        """
        text_lower = text.lower()

        # Count keyword matches for each category
        scores = {}
        for clause_type, keywords in CLAUSE_TYPES.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[clause_type] = score

        # Return the type with highest score, default to 'general'
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'general'

    def get_supported_file_types(self) -> List[str]:
        """Return list of supported file extensions"""
        return self.supported_extensions
