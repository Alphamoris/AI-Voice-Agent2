import pytest
import os
from unittest.mock import AsyncMock, MagicMock
from src.audio import AudioProcessor
from src.llm import LanguageModel
from src.speech import SpeechRecognizer
from src.voice import VoiceSynthesizer
from src.retell_agent import RetellAgent
from src.utils.session import SessionManager
from fastapi.testclient import TestClient
from app import app

@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables."""
    monkeypatch.setenv("RETELL_API_KEY", "test_retell_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test_deepgram_key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test_elevenlabs_key")

@pytest.fixture
def config():
    """Test configuration."""
    return {
        "audio": {
            "sample_rate": 16000,
            "channels": 1,
            "chunk_size": 1024,
            "buffer_size": 4096
        },
        "speech_recognition": {
            "default_provider": "deepgram",
            "providers": {
                "deepgram": {
                    "model": "nova-2",
                    "language": "en-US",
                    "interim_results": True,
                    "punctuate": True,
                    "diarize": True,
                    "smart_format": True
                }
            }
        },
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    "max_tokens": 150
                }
            }
        },
        "voice": {
            "default_provider": "elevenlabs",
            "providers": {
                "elevenlabs": {
                    "voice_id": "default",
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
        },
        "retell": {
            "enabled": True,
            "voice_id": "default",
            "language": "en-US",
            "stream_latency": 200,
            "use_enhanced_model": True,
            "auto_gain_control": True,
            "noise_suppression": True
        },
        "monitoring": {
            "log_level": "INFO",
            "metrics_enabled": True,
            "tracing_enabled": True
        },
        "security": {
            "token_expiry": 3600,
            "rate_limit": 100,
            "ip_whitelist": []
        }
    }

@pytest.fixture
def mock_openai(monkeypatch):
    """Mock OpenAI client."""
    class MockOpenAI:
        def __init__(self, api_key):
            self.chat = MagicMock()
            self.chat.completions = MagicMock()
            self.chat.completions.create = AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test response"))]
            ))
    monkeypatch.setattr("openai.AsyncOpenAI", MockOpenAI)

@pytest.fixture
def mock_deepgram(monkeypatch):
    """Mock Deepgram client."""
    class MockDeepgram:
        def __init__(self, api_key):
            pass
        async def transcription(self, *args, **kwargs):
            return {
                "results": {
                    "channels": [{
                        "alternatives": [{
                            "transcript": "Test transcript",
                            "confidence": 0.95
                        }]
                    }]
                }
            }
    monkeypatch.setattr("deepgram.Deepgram", MockDeepgram)

@pytest.fixture
def mock_elevenlabs(monkeypatch):
    """Mock ElevenLabs client."""
    def mock_generate(*args, **kwargs):
        return b"test_audio_data"
    monkeypatch.setattr("elevenlabs.generate", mock_generate)

@pytest.fixture
def audio_processor(config):
    """Audio processor fixture."""
    return AudioProcessor(config["audio"])

@pytest.fixture
def speech_recognizer(config):
    """Speech recognizer fixture."""
    return SpeechRecognizer(config["speech_recognition"])

@pytest.fixture
def language_model(config):
    """Language model fixture."""
    return LanguageModel(config["llm"])

@pytest.fixture
def voice_synthesizer(config):
    """Voice synthesizer fixture."""
    return VoiceSynthesizer(config["voice"])

@pytest.fixture
def session_manager(config):
    """Session manager fixture."""
    manager = SessionManager()
    manager.set_timeout(config["security"]["token_expiry"])
    return manager

@pytest.fixture
def retell_agent(config, session_manager):
    """Retell agent fixture."""
    return RetellAgent(config, session_manager)

@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)
