import pytest
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    # Check all components
    assert "status" in data
    assert data["status"] == "healthy"
    assert "components" in data
    
    components = data["components"]
    assert "audio_processor" in components
    assert "speech_recognition" in components
    assert "language_model" in components
    assert "voice_synthesis" in components
    assert "retell_agent" in components
    assert "sessions" in components

def test_websocket_connection(client):
    with client.websocket_connect("/conversation") as websocket:
        # Test connection is established
        websocket.send_bytes(b"test_audio_data")
        response = websocket.receive_bytes()
        assert isinstance(response, bytes)
