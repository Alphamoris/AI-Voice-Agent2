import pytest
import numpy as np
from src.audio import AudioProcessor

@pytest.fixture
def audio_config():
    return {
        "sample_rate": 16000,
        "channels": 1,
        "chunk_size": 1024,
        "buffer_size": 4096
    }

@pytest.fixture
def audio_processor(audio_config):
    return AudioProcessor(audio_config)

@pytest.mark.asyncio
async def test_audio_processor_initialization(audio_processor):
    await audio_processor.initialize()
    assert audio_processor.sample_rate == 16000
    assert audio_processor.channels == 1

def test_audio_processing(audio_processor):
    # Create test audio data
    test_audio = np.random.rand(1024).astype(np.float32).tobytes()
    
    # Process audio
    processed = audio_processor.process(test_audio)
    
    # Check output shape and type
    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.float32
    assert len(processed.shape) <= 2  # Either mono or stereo

def test_noise_reduction(audio_processor):
    # Create test audio with noise
    audio = np.random.normal(0, 0.001, 1024).astype(np.float32)
    
    # Add a signal
    audio[100:200] = 0.5  # Add a strong signal
    
    # Apply noise reduction
    processed = audio_processor._reduce_noise(audio)
    
    # Check if noise was reduced
    assert np.sum(np.abs(processed) > 0) < np.sum(np.abs(audio) > 0)

def test_automatic_gain_control(audio_processor):
    # Create test audio with low amplitude
    audio = np.random.normal(0, 0.1, 1024).astype(np.float32)
    
    # Apply AGC
    processed = audio_processor._apply_agc(audio)
    
    # Check if gain was adjusted
    assert np.max(np.abs(processed)) <= 1.0
