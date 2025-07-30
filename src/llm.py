
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from loguru import logger
import os

class LanguageModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get("default_provider", "openai")
        self.client: Optional[AsyncOpenAI] = None
        self.is_initialized = False
        self.conversation_history = {}
        
    async def initialize(self):
        """Initialize language model client."""
        try:
            if self.provider == "openai":
                self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.model = self.config["providers"]["openai"]["model"]
                self.temperature = self.config["providers"]["openai"]["temperature"]
                self.max_tokens = self.config["providers"]["openai"]["max_tokens"]
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            self.is_initialized = True
            logger.info(f"Language model initialized with provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise
        
    async def generate_response(self, user_input: str, session_id: str) -> str:
        """Generate response using the language model."""
        if not self.is_initialized:
            raise RuntimeError("Language model not initialized")

        try:
            if self.provider == "openai":
                # Get conversation history for this session
                history = self.conversation_history.get(session_id, [])
                
                # Prepare messages
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant engaged in a voice conversation. Keep your responses concise and natural."},
                ]
                
                # Add conversation history
                messages.extend(history)
                
                # Add user's input
                messages.append({"role": "user", "content": user_input})
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Update conversation history
                history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response_text}
                ])
                
                # Trim history if too long
                if len(history) > 10:  # Keep last 5 exchanges
                    history = history[-10:]
                
                self.conversation_history[session_id] = history
                
                return response_text
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
            
    def clear_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            
    def health_check(self) -> Dict:
        """Check the health of the language model service."""
        return {
            "status": "healthy" if self.is_initialized else "not_initialized",
            "provider": self.provider
        }
