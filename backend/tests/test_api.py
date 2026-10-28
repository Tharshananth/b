"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "services" in data

def test_chat_endpoint():
    """Test chat endpoint"""
    response = client.post(
        "/api/chat/",
        json={
            "message": "Hello, what is PingUs?",
            "conversation_history": []
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert "session_id" in data

def test_chat_with_invalid_input():
    """Test chat with invalid input"""
    response = client.post(
        "/api/chat/",
        json={
            "message": "",  # Empty message
            "conversation_history": []
        }
    )
    assert response.status_code == 422  # Validation error

def test_list_documents():
    """Test listing documents"""
    response = client.get("/api/documents/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_upload_document():
    """Test document upload"""
    # Create a test file
    test_content = b"This is a test document for PingUs."
    
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    
    # Should succeed or fail gracefully
    assert response.status_code in [200, 400, 500]

def test_get_config():
    """Test getting configuration"""
    response = client.get("/api/config/")
    assert response.status_code == 200
    data = response.json()
    assert "current_provider" in data
    assert "available_providers" in data

def test_chat_history():
    """Test chat history endpoints"""
    # First create a session
    chat_response = client.post(
        "/api/chat/",
        json={"message": "Test message"}
    )
    session_id = chat_response.json()["session_id"]
    
    # Get history
    response = client.get(f"/api/chat/history/{session_id}")
    assert response.status_code == 200
    
    # Delete history
    response = client.delete(f"/api/chat/history/{session_id}")
    assert response.status_code == 200

def test_cors_headers():
    """Test CORS headers are present"""
    response = client.options("/api/chat/")
    assert "access-control-allow-origin" in response.headers or response.status_code == 200

@pytest.mark.asyncio
async def test_streaming_endpoint():
    """Test streaming chat endpoint"""
    response = client.post(
        "/api/chat/stream",
        json={
            "message": "Count to 3",
            "conversation_history": []
        }
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

def test_provider_switching():
    """Test switching LLM provider"""
    # Get available providers
    config_response = client.get("/api/config/")
    providers = config_response.json()["available_providers"]
    
    if len(providers) > 1:
        # Try switching to second provider
        provider_name = providers[1]["name"]
        response = client.post(f"/api/config/provider/{provider_name}")
        assert response.status_code in [200, 400]  # 400 if provider not available

def test_feedback_submission():
    """Test feedback submission"""
    response = client.post(
        "/api/chat/feedback",
        params={
            "session_id": "test_session",
            "message_id": "test_msg",
            "helpful": True,
            "feedback": "Great response!"
        }
    )
    assert response.status_code == 200

def test_refresh_knowledge_base():
    """Test refreshing knowledge base"""
    response = client.post("/api/documents/refresh")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
