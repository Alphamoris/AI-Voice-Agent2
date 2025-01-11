import asyncio
import websockets
import sounddevice as sd
import numpy as np
from typing import Optional
import click
import sys
import json
from loguru import logger

async def record_audio(duration: float = 0.5, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        return audio_data
    except Exception as e:
        logger.error(f"Error recording audio: {str(e)}")
        raise

async def play_audio(audio_data: bytes, sample_rate: int = 16000):
    """Play audio data."""
    try:
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        sd.play(audio_array, sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"Error playing audio: {str(e)}")
        raise

async def main(server_url: str = "ws://localhost:8000/conversation"):
    """Main client function."""
    try:
        logger.info(f"Connecting to server at {server_url}")
        async with websockets.connect(server_url) as websocket:
            logger.info("Connected to voice agent server")
            logger.info("Press Enter to start speaking (q + Enter to quit)")
            
            while True:
                command = input()
                if command.lower() == 'q':
                    break
                
                try:
                    logger.info("Recording... (speak now)")
                    audio_data = await record_audio()
                    
                    # Send audio to server
                    await websocket.send(audio_data.tobytes())
                    
                    # Receive response
                    logger.info("Waiting for response...")
                    response = await websocket.recv()
                    
                    # Play response
                    logger.info("Playing response...")
                    await play_audio(response)
                except Exception as e:
                    logger.error(f"Error during conversation: {str(e)}")
                    continue
                
    except websockets.exceptions.ConnectionClosed:
        logger.error("Connection to server closed")
    except Exception as e:
        logger.error(f"Error: {str(e)}")

@click.command()
@click.option('--server', default="ws://localhost:8000/conversation", help='WebSocket server URL')
def run(server):
    """Run the test client."""
    try:
        asyncio.run(main(server))
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    run()
