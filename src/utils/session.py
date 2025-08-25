








from datetime import datetime, timedelta
from typing import Dict, Optional
import uuid
from loguru import logger

class Session:
    def __init__(self, id: str):
        self.id = id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = True

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()

class SessionManager:
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = 3600  # Default timeout of 1 hour
        self.is_initialized = True
        logger.info("Session manager initialized")

    def set_timeout(self, timeout: int):
        """Set session timeout in seconds."""
        self.session_timeout = timeout

    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = Session(session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        session = self.sessions.get(session_id)
        if session and session.is_active:
            session.update_activity()
            return session
        return None

    def end_session(self, session_id: str) -> bool:
        """End a session."""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            del self.sessions[session_id]
            logger.info(f"Removed inactive session: {session_id}")
            return True
        return False

    def cleanup_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.now()
        timeout = timedelta(seconds=self.session_timeout)
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if current_time - session.last_activity > timeout
        ]
        for session_id in expired_sessions:
            self.end_session(session_id)

    def get_active_sessions_count(self) -> int:
        """Return the number of active sessions."""
        return sum(1 for session in self.sessions.values() if session.is_active)
