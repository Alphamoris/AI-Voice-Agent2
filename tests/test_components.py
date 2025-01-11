import pytest
import numpy as np
from src.utils.exceptions import VoiceAgentError

@pytest.mark.asyncio
async def test_audio_processor(audio_processor):
    """Test audio processor functionality."""
    await audio_processor.initialize()
    assert audio_processor.is_initialized

    # Test audio processing
    test_audio = np.random.rand(1024).astype(np.float32).tobytes()
    processed = audio_processor.process(test_audio)
    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.float32

    # Test noise reduction
    noisy_audio = np.random.normal(0, 0.001, 1024).astype(np.float32)
    noisy_audio[100:200] = 0.5  # Add a strong signal
    processed = audio_processor._reduce_noise(noisy_audio)
    assert np.max(np.abs(processed)) <= 1.0

    # Test automatic gain control
    quiet_audio = np.random.normal(0, 0.1, 1024).astype(np.float32)
    processed = audio_processor._apply_agc(quiet_audio)
    assert np.max(np.abs(processed)) <= 1.0

@pytest.mark.asyncio
async def test_speech_recognition(speech_recognizer):
    """Test speech recognition functionality."""
    await speech_recognizer.initialize()
    assert speech_recognizer.is_initialized

    # Test transcription
    audio_data = np.random.rand(16000).astype(np.float32)
    result = await speech_recognizer.transcribe(audio_data)
    assert isinstance(result.text, str)
    assert result.is_final
    assert result.language == "en-US"

@pytest.mark.asyncio
async def test_language_model(language_model):
    """Test language model functionality."""
    await language_model.initialize()
    assert language_model.is_initialized

    # Test response generation
    response = await language_model.generate_response("Hello!", "test_session")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "test_session" in language_model.conversation_history

@pytest.mark.asyncio
async def test_voice_synthesis(voice_synthesizer):
    """Test voice synthesis functionality."""
    await voice_synthesizer.initialize()
    assert voice_synthesizer.is_initialized

    # Test synthesis
    audio_data = await voice_synthesizer.synthesize("Hello, world!")
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0

@pytest.mark.asyncio
async def test_retell_agent(retell_agent):
    """Test Retell agent functionality."""
    await retell_agent.initialize()
    assert retell_agent.is_initialized

    # Test audio handling
    class MockWebSocket:
        async def send(self, data):
            assert isinstance(data, str)
    
    audio_data = b"test_audio_data"
    await retell_agent.handle_audio(MockWebSocket(), audio_data)

    # Test message handling
    message = {
        "type": "transcription",
        "data": {
            "text": "Hello",
            "is_final": True
        }
    }
    response = await retell_agent.handle_message(message)
    assert isinstance(response, bytes)

@pytest.mark.asyncio
async def test_session_manager(session_manager):
    """Test session manager functionality."""
    # Test session creation
    session_id = session_manager.create_session()
    assert session_id in session_manager.sessions

    # Test session retrieval
    session = session_manager.get_session(session_id)
    assert session.id == session_id
    assert session.created_at is not None

    # Test session cleanup
    session_manager.cleanup_sessions()
    assert session_id in session_manager.sessions  # Recent session should remain

def test_error_handling():
    """Test error handling."""
    # Test VoiceAgentError
    with pytest.raises(VoiceAgentError):
        raise VoiceAgentError("Test error")
