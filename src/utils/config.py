








import os
import yaml
from loguru import logger
from typing import Dict, Any
from pathlib import Path
from src.utils.exceptions import ConfigurationError

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        # Use default config path if not provided
        if not config_path:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")

        # Load configuration file
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate configuration
        validate_config(config)

        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration structure."""
    required_sections = [
        "app",
        "audio",
        "speech_recognition",
        "llm",
        "voice",
        "retell",
        "monitoring",
        "security"
    ]

    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required section: {section}")

    # Validate audio configuration
    audio_config = config["audio"]
    required_audio_params = ["sample_rate", "channels", "chunk_size", "buffer_size"]
    for param in required_audio_params:
        if param not in audio_config:
            raise ConfigurationError(f"Missing required audio parameter: {param}")

    # Validate speech recognition configuration
    speech_config = config["speech_recognition"]
    if "default_provider" not in speech_config or "providers" not in speech_config:
        raise ConfigurationError("Invalid speech recognition configuration")

    # Validate language model configuration
    llm_config = config["llm"]
    if "default_provider" not in llm_config or "providers" not in llm_config:
        raise ConfigurationError("Invalid language model configuration")

    # Validate voice configuration
    voice_config = config["voice"]
    if "default_provider" not in voice_config or "providers" not in voice_config:
        raise ConfigurationError("Invalid voice configuration")

    # Validate Retell configuration
    retell_config = config["retell"]
    required_retell_params = ["enabled", "voice_id", "language", "stream_latency"]
    for param in required_retell_params:
        if param not in retell_config:
            raise ConfigurationError(f"Missing required Retell parameter: {param}")

    # Validate monitoring configuration
    monitoring_config = config["monitoring"]
    if "log_level" not in monitoring_config:
        raise ConfigurationError("Missing log level in monitoring configuration")

    # Validate security configuration
    security_config = config["security"]
    if "token_expiry" not in security_config:
        raise ConfigurationError("Missing token expiry in security configuration")
