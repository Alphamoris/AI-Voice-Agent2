import pytest
from unittest.mock import AsyncMock, MagicMock
import json
from src.retell_agent import RetellAgent
from src.utils.session import SessionManager

@pytest.fixture
def config():
    return {
        "retell": {
            "enabled": True,
            "voice_id": "default",
            "language": "en-US",
            "stream_latency": 200,
            "use_enhanced_model": True,
            "auto_gain_control": True,
            "noise_suppression": True
        },
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
                    "punctuate": True
                }
            }
        },
        "llm": {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            }
        },
        "voice": {
            "default_provider": "elevenlabs",
            "providers": {
                "elevenlabs": {
                    "voice_id": "default"
                }
            }
        },
        "security": {
            "token_expiry": 3600
        }
    }

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def retell_agent(config, session_manager):
    agent = RetellAgent(config, session_manager)
    return agent

@pytest.mark.asyncio
async def test_retell_agent_initialization(retell_agent):
    assert retell_agent.config is not None
    assert retell_agent.session_manager is not None

@pytest.mark.asyncio
async def test_handle_audio(retell_agent):
    # Create a mock websocket
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    
    # Test audio handling
    audio_data = b"test_audio_data"
    session_id = retell_agent.session_manager.create_session()
    
    await retell_agent.handle_audio(mock_ws, audio_data, session_id)
    
    # Verify the websocket send was called
    mock_ws.send.assert_called()
    # Get the call arguments
    call_args = mock_ws.send.call_args[0][0]
    # Parse the JSON message
    message = json.loads(call_args)
    # Verify message structure
    assert message["type"] == "audio"
    assert isinstance(message["data"], str)

@pytest.mark.asyncio
async def test_handle_message(retell_agent):
    # Create a mock websocket
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    
    # Test message handling
    test_message = "Hello, how are you?"
    session_id = retell_agent.session_manager.create_session()
    
    await retell_agent.handle_message(mock_ws, test_message, session_id)
    
    # Verify the websocket send was called
    mock_ws.send.assert_called()
    # Get the call arguments
    call_args = mock_ws.send.call_args[0][0]
    # Parse the JSON message
    message = json.loads(call_args)
    # Verify message structure
    assert message["type"] == "text"
    assert isinstance(message["data"], str)
