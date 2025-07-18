from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_endpoint():
    payload = {"query": "46M, knee surgery, Pune, 3-month policy"}
    r = client.post("/query", json=payload)
    assert r.status_code == 200
    assert "decision" in r.json()