









import numpy as np
from typing import Dict, Optional
import sounddevice as sd
from loguru import logger

class AudioProcessor:
    def __init__(self, config: Dict):
        self.sample_rate = config["sample_rate"]
        self.channels = config["channels"]
        self.chunk_size = config["chunk_size"]
        self.buffer_size = config["buffer_size"]
        
        # Initialize audio buffer
        self.buffer = np.zeros((self.buffer_size, self.channels))
        self.buffer_index = 0
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize audio processor."""
        try:
            # Initialize sounddevice with configured settings
            sd.default.samplerate = self.sample_rate
            sd.default.channels = self.channels
            self.is_initialized = True
            logger.info("Audio processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio processor: {str(e)}")
            raise
            
    def process(self, audio_data: bytes) -> np.ndarray:
        """Process incoming audio data."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Reshape if stereo
            if self.channels == 2:
                audio_array = audio_array.reshape(-1, 2)
            
            # Apply noise reduction
            processed_audio = self._reduce_noise(audio_array)
            
            # Apply automatic gain control
            processed_audio = self._apply_agc(processed_audio)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise
            
    def _reduce_noise(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply basic noise reduction."""
        # Simple noise gate
        noise_threshold = 0.01
        # Create a copy to avoid modifying read-only array
        audio_array = audio_array.copy()
        audio_array[np.abs(audio_array) < noise_threshold] = 0
        return audio_array
        
    def _apply_agc(self, audio_array: np.ndarray) -> np.ndarray:
        """Apply automatic gain control."""
        # Create a copy to avoid modifying read-only array
        audio_array = audio_array.copy()
        max_amplitude = np.max(np.abs(audio_array))
        if max_amplitude > 0:
            gain = 0.7 / max_amplitude
            audio_array *= gain
        return audio_array
        
    def get_stream_parameters(self) -> Dict:
        """Return audio stream parameters."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size
        }
        
    def reset(self):
        """Reset audio processor state."""
        self.buffer = np.zeros((self.buffer_size, self.channels))
        self.buffer_index = 0
