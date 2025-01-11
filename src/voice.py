from typing import Dict, Optional
import elevenlabs
import numpy as np
from loguru import logger
import os

class VoiceSynthesizer:
    def __init__(self, config: Dict):
        self.config = config
        self.provider = config["default_provider"]
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize voice synthesis clients."""
        try:
            self.provider = self.config.get("default_provider", "elevenlabs")
            provider_config = self.config["providers"][self.provider]
            
            # Initialize ElevenLabs client
            elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))
            self.voice_id = provider_config.get("voice_id", "default")
            self.stability = provider_config.get("stability", 0.5)
            self.similarity_boost = provider_config.get("similarity_boost", 0.75)
            
            self.is_initialized = True
            logger.info(f"Voice synthesizer initialized with provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize voice synthesizer: {str(e)}")
            raise
        
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to speech."""
        if not self.is_initialized:
            raise RuntimeError("Voice synthesizer not initialized")
            
        try:
            if self.provider == "elevenlabs":
                return await self._synthesize_elevenlabs(text)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Voice synthesis error: {str(e)}")
            raise
            
    async def _synthesize_elevenlabs(self, text: str) -> bytes:
        """Synthesize text using ElevenLabs API."""
        try:
            # Get voice configuration
            voice_config = self.config["providers"]["elevenlabs"]
            
            # Generate audio
            audio = elevenlabs.generate(
                text=text,
                voice=voice_config["voice_id"],
                model="eleven_monolingual_v1",
                stability=voice_config["stability"],
                similarity_boost=voice_config["similarity_boost"]
            )
            
            return audio
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {str(e)}")
            raise
            
    def health_check(self) -> Dict:
        """Check the health of the voice synthesis service."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "provider": self.provider
        }
