import pytest
import numpy as np
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def client(mock_env, mock_openai, mock_deepgram, mock_elevenlabs):
    """Test client fixture."""
    return TestClient(app)

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "components" in data

@pytest.mark.asyncio
async def test_full_conversation_flow(
    audio_processor,
    speech_recognizer,
    language_model,
    voice_synthesizer,
    retell_agent
):
    """Test the full conversation flow."""
    # Initialize components
    await audio_processor.initialize()
    await speech_recognizer.initialize()
    await language_model.initialize()
    await voice_synthesizer.initialize()
    await retell_agent.initialize()

    # Create test audio data
    audio_data = np.random.rand(16000).astype(np.float32)

    # Process audio
    processed_audio = audio_processor.process(audio_data.tobytes())
    assert isinstance(processed_audio, np.ndarray)

    # Transcribe audio
    transcription = await speech_recognizer.transcribe(processed_audio)
    assert isinstance(transcription.text, str)
    assert transcription.is_final

    # Generate response
    response = await language_model.generate_response(transcription.text, "test_session")
    assert isinstance(response, str)
    assert len(response) > 0

    # Synthesize speech
    audio_response = await voice_synthesizer.synthesize(response)
    assert isinstance(audio_response, bytes)
    assert len(audio_response) > 0

@pytest.mark.asyncio
async def test_websocket_connection(client):
    """Test WebSocket connection."""
    with client.websocket_connect("/conversation") as websocket:
        # Send test audio data
        test_data = np.random.rand(1024).astype(np.float32).tobytes()
        websocket.send_bytes(test_data)

        # Receive response
        response = websocket.receive_bytes()
        assert isinstance(response, bytes)
        assert len(response) > 0
