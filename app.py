import os
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import yaml
from src.utils.config import load_config
from src.utils.session import SessionManager
from src.retell_agent import RetellAgent

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = load_config()

# Initialize session manager
session_manager = SessionManager()
session_manager.set_timeout(config["security"]["token_expiry"])

# Initialize Retell agent
retell_agent = RetellAgent(config, session_manager)

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        await retell_agent.initialize()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if all components are initialized
        if not retell_agent.is_initialized:
            return {"status": "unhealthy", "message": "Components not initialized"}

        # Check if required API keys are set
        required_keys = ["OPENAI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        
        if missing_keys:
            return {
                "status": "unhealthy",
                "message": f"Missing API keys: {', '.join(missing_keys)}"
            }

        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "message": str(e)}

@app.websocket("/conversation")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for voice conversations."""
    try:
        # Accept connection
        await websocket.accept()
        session_id = str(uuid.uuid4())
        logger.info(f"New conversation session started: {session_id}")

        while True:
            # Receive message
            message = await websocket.receive()

            if message.get("type") == "bytes":
                # Handle audio data
                audio_data = message.get("bytes")
                transcription = await retell_agent.handle_audio(websocket, audio_data)

                if transcription and transcription.get("is_final"):
                    # Generate and send response
                    response = await retell_agent.handle_message({
                        "type": "transcription",
                        "data": transcription
                    }, session_id)

                    if response:
                        await websocket.send_json(response)

            elif message.get("type") == "text":
                # Handle text messages
                data = message.get("text")
                response = await retell_agent.handle_message({
                    "type": "text",
                    "data": data
                }, session_id)

                if response:
                    await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed: {session_id}")
        session_manager.end_session(session_id)
    except Exception as e:
        logger.error(f"Error in websocket endpoint: {str(e)}")
        try:
            await websocket.close()
        except:
            pass
        session_manager.end_session(session_id)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        await retell_agent.cleanup()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add(
        "logs/voice_agent.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )
