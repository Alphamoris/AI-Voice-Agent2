


from typing import Dict, Optional
import websockets
import json
import asyncio
import os
from loguru import logger
from src.audio import AudioProcessor
from src.speech import SpeechRecognizer
from src.llm import LanguageModel
from src.voice import VoiceSynthesizer
from src.utils.session import SessionManager

class RetellAgent:
    def __init__(self, config: Dict, session_manager: SessionManager):
        self.config = config["retell"]
        self.session_manager = session_manager
        self.audio_processor = AudioProcessor(config["audio"])
        self.speech_recognizer = SpeechRecognizer(config["speech_recognition"])
        self.language_model = LanguageModel(config["llm"])
        self.voice_synthesizer = VoiceSynthesizer(config["voice"])
        self.api_key = os.getenv("RETELL_API_KEY")
        self.is_initialized = False

    async def initialize(self):
        """Initialize all components."""
        try:
            await self.audio_processor.initialize()
            await self.speech_recognizer.initialize()
            await self.language_model.initialize()
            await self.voice_synthesizer.initialize()
            self.is_initialized = True
            logger.info("Retell agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Retell agent: {str(e)}")
            raise

    async def start_conversation(self, session_id: str):
        """Start a new conversation session with Retell."""
        if not self.is_initialized:
            raise RuntimeError("Retell agent not initialized")

        try:
            # Connect to Retell websocket
            websocket_url = f"wss://api.retellai.com/websocket?api_key={self.api_key}"
            async with websockets.connect(websocket_url) as websocket:
                # Configure conversation settings
                config = {
                    "voice_id": self.config["voice_id"],
                    "language": self.config["language"],
                    "stream_latency_ms": self.config["stream_latency"],
                    "use_enhanced_model": self.config["use_enhanced_model"],
                    "audio_config": {
                        "auto_gain_control": self.config["auto_gain_control"],
                        "noise_suppression": self.config["noise_suppression"]
                    }
                }
                
                await websocket.send(json.dumps({
                    "type": "config",
                    "data": config
                }))

                return websocket

        except Exception as e:
            logger.error(f"Failed to start Retell conversation: {str(e)}")
            raise

    async def handle_audio(self, websocket, audio_data: bytes):
        """Handle incoming audio data."""
        try:
            # Process audio
            processed_audio = self.audio_processor.process(audio_data)
            
            # Get transcription
            transcription = await self.speech_recognizer.transcribe(processed_audio)
            
            # Send transcription back to client
            await websocket.send_json({
                "type": "transcription",
                "data": transcription
            })
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error handling audio: {str(e)}")
            raise

    async def handle_message(self, message: Dict, session_id: str):
        """Handle incoming message from Retell."""
        try:
            if message["type"] == "transcription":
                text = message["data"]["text"]
                is_final = message["data"]["is_final"]

                if is_final and text:
                    # Generate response using language model
                    response = await self.language_model.generate_response(text, session_id)
                    
                    # Synthesize speech
                    audio_data = await self.voice_synthesizer.synthesize(response)
                    
                    return {
                        "type": "response",
                        "data": {
                            "text": response,
                            "audio": audio_data
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.audio_processor.cleanup()
            await self.speech_recognizer.cleanup()
            await self.language_model.cleanup()
            await self.voice_synthesizer.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
