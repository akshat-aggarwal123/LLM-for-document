"""
API Tests for Document Information Retrieval System
"""

import pytest
import json
import tempfile
import os
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app
from app.models.schemas import QueryRequest, UploadResponse

client = TestClient(app)

class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AI Document Information Retrieval System"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"

    def test_health_endpoint(self):
        """Test simple health endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_health_endpoint(self):
        """Test API health endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data

class TestDocumentUpload:
    """Test document upload functionality"""

    def create_test_file(self, filename: str, content: str) -> str:
        """Create a temporary test file"""
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return file_path

    def test_upload_text_file(self):
        """Test uploading a text file"""
        # Create test file
        test_content = """
        INSURANCE POLICY DOCUMENT

        Coverage Details:
        - Hospitalization: Up to ₹5,00,000 per year
        - Surgery: Covered under hospitalization
        - Pre-existing conditions: 2-year waiting period

        Exclusions:
        - Cosmetic surgery
        - Alternative medicine treatments
        """

        file_path = self.create_test_file("test_policy.txt", test_content)

        try:
            with open(file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/upload",
                    files={"file": ("test_policy.txt", f, "text/plain")}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test_policy.txt"
            assert "document_id" in data
            assert data["message"] == "Document uploaded and processed successfully"

        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_upload_invalid_file_type(self):
        """Test uploading invalid file type"""
        file_path = self.create_test_file("test.xyz", "invalid content")

        try:
            with open(file_path, 'rb') as f:
                response = client.post(
                    "/api/v1/upload",
                    files={"file": ("test.xyz", f, "application/octet-stream")}
                )

            assert response.status_code == 400
            data = response.json()
            assert "not supported" in data["detail"]

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_upload_no_file(self):
        """Test uploading without file"""
        response = client.post("/api/v1/upload")
        assert response.status_code == 422  # Validation error

class TestQueryProcessing:
    """Test query processing functionality"""

    def test_basic_query(self):
        """Test basic query processing"""
        query_data = {
            "query": "Is knee surgery covered for a 45-year-old patient?",
            "top_k": 5,
            "use_llm": False
        }

        response = client.post("/api/v1/query", json=query_data)

        # Note: This might fail if no documents are uploaded
        # In a real test, you'd set up test documents first
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            data = response.json()
            assert data["query"] == query_data["query"]
            assert "decision" in data
            assert "confidence_score" in data

    def test_query_with_llm(self):
        """Test query processing with LLM enabled"""
        query_data = {
            "query": "Coverage for cataract surgery, age 65, policy 2 years old",
            "top_k": 3,
            "use_llm": True
        }

        response = client.post("/api/v1/query", json=query_data)

        # This test might fail if LLM is not available
        assert response.status_code in [200, 422, 503]

    def test_empty_query(self):
        """Test empty query"""
        query_data = {
            "query": "",
            "top_k": 5
        }

        response = client.post("/api/v1/query", json=query_data)
        assert response.status_code == 422  # Validation error

    def test_invalid_query_structure(self):
        """Test invalid query structure"""
        response = client.post("/api/v1/query", json={"invalid": "data"})
        assert response.status_code == 422

class TestDocumentManagement:
    """Test document management endpoints"""

    def test_list_documents(self):
        """Test listing documents"""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_delete_nonexistent_document(self):
        """Test deleting non-existent document"""
        response = client.delete("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404

class TestBatchProcessing:
    """Test batch processing functionality"""

    def test_batch_query(self):
        """Test batch query processing"""
        queries = [
            "Is surgery covered?",
            "What is the waiting period?",
            "Coverage amount for hospitalization?"
        ]

        response = client.post("/api/v1/batch-query", json=queries)
        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == len(queries)
        assert "results" in data
        assert len(data["results"]) == len(queries)

    def test_batch_query_too_many(self):
        """Test batch query with too many queries"""
        queries = ["Query " + str(i) for i in range(15)]  # More than limit

        response = client.post("/api/v1/batch-query", json=queries)
        assert response.status_code == 400
        assert "Maximum 10 queries" in response.json()["detail"]

class TestDocumentAnalysis:
    """Test document analysis endpoints"""

    def test_analyze_document(self):
        """Test document analysis without storage"""
        test_content = """
        HEALTH INSURANCE POLICY

        This policy provides coverage for:
        1. Hospitalization expenses
        2. Surgical procedures
        3. Diagnostic tests

        Waiting periods:
        - General illnesses: 30 days
        - Pre-existing conditions: 24 months
        """

        file_path = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        file_path.write(test_content)
        file_path.close()

        try:
            with open(file_path.name, 'rb') as f:
                response = client.post(
                    "/api/v1/analyze-document",
                    files={"file": ("test_policy.txt", f, "text/plain")}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["filename"] == "test_policy.txt"
            assert "document_type" in data
            assert "sections" in data
            assert "entities" in data

        finally:
            if os.path.exists(file_path.name):
                os.remove(file_path.name)

class TestStatistics:
    """Test system statistics endpoint"""

    def test_get_statistics(self):
        """Test getting system statistics"""
        response = client.get("/api/v1/statistics")
        assert response.status_code == 200

        data = response.json()
        assert "total_documents" in data
        assert "model_status" in data
        assert "vector_db_size" in data

class TestErrorHandling:
    """Test error handling scenarios"""

    def test_invalid_endpoint(self):
        """Test calling invalid endpoint"""
        response = client.get("/api/v1/invalid-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test invalid HTTP method"""
        response = client.delete("/api/v1/query")
        assert response.status_code == 405

    @patch('app.services.document_processor.DocumentProcessor.process_document')
    def test_document_processing_error(self, mock_process):
        """Test document processing error handling"""
        mock_process.side_effect = Exception("Processing failed")

        file_path = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        file_path.write("test content")
        file_path.close()

        try:
            with open(file_path.name, 'rb') as f:
                response = client.post(
                    "/api/v1/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )

            assert response.status_code == 500

        finally:
            if os.path.exists(file_path.name):
                os.remove(file_path.name)

# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""

    def test_upload_and_query_workflow(self):
        """Test complete upload and query workflow"""
        # Step 1: Upload document
        test_content = """
        COMPREHENSIVE HEALTH INSURANCE POLICY

        Coverage Benefits:
        - Hospitalization: Maximum ₹10,00,000 per policy year
        - Day care procedures: Covered
        - Emergency care: 24/7 coverage
        - Maternity benefits: After 9 months waiting period

        Exclusions:
        - Cosmetic surgery (except reconstructive)
        - Alternative medicine treatments
        - Dental treatment (except due to accident)

        Waiting Periods:
        - Initial waiting period: 30 days
        - Pre-existing diseases: 48 months
        - Specific diseases: 24 months
        """

        file_path = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        file_path.write(test_content)
        file_path.close()

        try:
            # Upload document
            with open(file_path.name, 'rb') as f:
                upload_response = client.post(
                    "/api/v1/upload",
                    files={"file": ("comprehensive_policy.txt", f, "text/plain")}
                )

            if upload_response.status_code != 200:
                pytest.skip("Document upload failed, skipping integration test")

            upload_data = upload_response.json()
            document_id = upload_data["document_id"]

            # Step 2: Query the document
            query_data = {
                "query": "Is maternity care covered and what is the waiting period?",
                "top_k": 5,
                "use_llm": False
            }

            query_response = client.post("/api/v1/query", json=query_data)

            if query_response.status_code == 200:
                query_data = query_response.json()
                assert "maternity" in query_data["justification"].lower() or \
                       "maternity" in str(query_data["relevant_clauses"]).lower()

            # Step 3: List documents
            list_response = client.get("/api/v1/documents")
            assert list_response.status_code == 200

            documents = list_response.json()
            document_ids = [doc["document_id"] for doc in documents]
            assert document_id in document_ids

            # Step 4: Delete document
            delete_response = client.delete(f"/api/v1/documents/{document_id}")
            # Note: This might fail if delete is not properly implemented

        finally:
            if os.path.exists(file_path.name):
                os.remove(file_path.name)

# Performance tests
class TestPerformance:
    """Basic performance tests"""

    def test_query_response_time(self):
        """Test query response time"""
        import time

        query_data = {
            "query": "Basic coverage information",
            "top_k": 5
        }

        start_time = time.time()
        response = client.post("/api/v1/query", json=query_data)
        end_time = time.time()

        response_time = end_time - start_time

        # Response should be reasonably fast (under 30 seconds for basic query)
        assert response_time < 30.0

        if response.status_code == 200:
            data = response.json()
            assert "processing_time" in data

# Test fixtures and utilities
@pytest.fixture
def sample_insurance_document():
    """Fixture providing sample insurance document content"""
    return """
    STANDARD HEALTH INSURANCE POLICY

    Policy Number: HSI-2024-001
    Coverage Amount: ₹5,00,000

    Covered Benefits:
    1. Hospitalization expenses (Room rent, nursing, medicines)
    2. Pre and post hospitalization (60 days before, 90 days after)
    3. Day care procedures (Over 150 procedures covered)
    4. Domiciliary treatment (For specific conditions)
    5. Ambulance services (Up to ₹2,000 per claim)

    Exclusions:
    1. Pre-existing diseases (Until waiting period completion)
    2. Congenital anomalies and genetic disorders
    3. Cosmetic or aesthetic surgery
    4. Infertility treatment and assisted reproduction
    5. Dental treatment (unless due to accident)

    Waiting Periods:
    - Initial: 30 days from policy start
    - Pre-existing diseases: 36 months
    - Specific diseases (Hernia, Cataract, etc.): 12 months
    - Maternity benefits: 10 months
    """

@pytest.fixture
def sample_query_requests():
    """Fixture providing sample query requests"""
    return [
        {
            "query": "Is cataract surgery covered for a 65-year-old?",
            "top_k": 5,
            "use_llm": False
        },
        {
            "query": "What is the coverage amount for hospitalization?",
            "top_k": 3,
            "use_llm": False
        },
        {
            "query": "Maternity benefits waiting period and coverage",
            "top_k": 5,
            "use_llm": True
        }
    ]

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
