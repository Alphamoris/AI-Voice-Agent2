import pytest
from src.llm import LanguageModel

@pytest.fixture
def llm_config():
    return {
        "default_provider": "openai",
        "providers": {
            "openai": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
    }

@pytest.fixture
def language_model(llm_config):
    return LanguageModel(llm_config)

@pytest.mark.asyncio
async def test_llm_initialization(language_model, monkeypatch):
    # Mock OpenAI client
    class MockOpenAI:
        def __init__(self, api_key):
            pass
    
    monkeypatch.setattr("openai.OpenAI", MockOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    
    await language_model.initialize()
    assert language_model.provider == "openai"
    assert language_model.is_initialized

@pytest.mark.asyncio
async def test_response_generation(language_model, monkeypatch):
    # Mock OpenAI response
    async def mock_generate(*args, **kwargs):
        return "This is a test response."
    
    # Mock initialization
    class MockOpenAI:
        def __init__(self, api_key):
            pass
    
    monkeypatch.setattr("openai.OpenAI", MockOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    await language_model.initialize()
    
    monkeypatch.setattr(language_model, "_generate_openai_response", mock_generate)
    
    # Test response generation
    response = await language_model.generate_response("Hello!", "test_session")
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Check conversation history
    assert "test_session" in language_model.conversation_history
    assert len(language_model.conversation_history["test_session"]) > 0
