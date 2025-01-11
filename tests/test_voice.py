import pytest
from src.voice import VoiceSynthesizer

@pytest.fixture
def voice_config():
    return {
        "default_provider": "elevenlabs",
        "providers": {
            "elevenlabs": {
                "voice_id": "default",
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }
    }

@pytest.fixture
def voice_synthesizer(voice_config):
    return VoiceSynthesizer(voice_config)

@pytest.mark.asyncio
async def test_voice_synthesizer_initialization(voice_synthesizer):
    await voice_synthesizer.initialize()
    assert voice_synthesizer.provider == "elevenlabs"
    assert voice_synthesizer.is_initialized

@pytest.mark.asyncio
async def test_voice_synthesis(voice_synthesizer, monkeypatch):
    # Mock ElevenLabs response
    async def mock_synthesize(*args, **kwargs):
        # Return dummy audio bytes
        return b"dummy_audio_data"
    
    monkeypatch.setattr(voice_synthesizer, "_synthesize_elevenlabs", mock_synthesize)
    
    # Test synthesis
    audio_data = await voice_synthesizer.synthesize("Hello, world!")
    assert isinstance(audio_data, bytes)
    assert len(audio_data) > 0
