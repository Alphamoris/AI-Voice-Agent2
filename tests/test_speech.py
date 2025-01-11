import pytest
import numpy as np
from src.speech import SpeechRecognizer

@pytest.fixture
def speech_config():
    return {
        "default_provider": "deepgram",
        "providers": {
            "deepgram": {
                "model": "nova-2",
                "language": "en-US",
                "interim_results": True,
                "punctuate": True,
                "diarize": True
            }
        }
    }

@pytest.fixture
def speech_recognizer(speech_config):
    return SpeechRecognizer(speech_config)

@pytest.mark.asyncio
async def test_speech_recognizer_initialization(speech_recognizer):
    await speech_recognizer.initialize()
    assert speech_recognizer.provider == "deepgram"
    assert speech_recognizer.is_initialized

@pytest.mark.asyncio
async def test_transcription(speech_recognizer, monkeypatch):
    # Mock Deepgram response
    async def mock_transcribe(*args, **kwargs):
        return {
            "text": "Hello, world!",
            "is_final": True,
            "confidence": 0.95,
            "language": "en-US"
        }
    
    monkeypatch.setattr(speech_recognizer, "_transcribe_deepgram", mock_transcribe)
    
    # Create test audio data
    audio_data = np.random.rand(16000).astype(np.float32)
    
    # Test transcription
    result = await speech_recognizer.transcribe(audio_data)
    assert result.text == "Hello, world!"
    assert result.is_final
    assert result.confidence > 0.9
    assert result.language == "en-US"
