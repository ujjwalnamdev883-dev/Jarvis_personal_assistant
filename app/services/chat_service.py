# app/services/chat_service.py

import json
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# PROJECT CONFIGURATION
# -------------------------------------------------------------------------

CHATS_DATA_DIR = Path("database/chats_data")
CHATS_DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHAT_HISTORY_TURNS = 20

VALID_ROLES = {"user", "assistant"}


# -------------------------------------------------------------------------
# MESSAGE MODEL
# -------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str


# -------------------------------------------------------------------------
# PLACEHOLDER SERVICES
# Replace these with your actual implementations later
# -------------------------------------------------------------------------

class GroqService:
    def get_response(
        self,
        question: str,
        chat_history: List[Tuple[str, str]]
    ) -> str:
        return "Groq AI Response Placeholder"


class RealtimeGroqService:
    def get_response(
        self,
        question: str,
        chat_history: List[Tuple[str, str]]
    ) -> str:
        return "Realtime Tavily + Groq Response Placeholder"


# -------------------------------------------------------------------------
# MAIN CHAT SERVICE
# -------------------------------------------------------------------------

class ChatService:

    def __init__(
        self,
        groq_service: GroqService,
        realtime_service: Optional[RealtimeGroqService] = None
    ):
        self.groq_service = groq_service
        self.realtime_service = realtime_service

        # session_id -> List[ChatMessage]
        self.sessions: Dict[str, List[ChatMessage]] = {}

    # ---------------------------------------------------------------------
    # SESSION HELPERS
    # ---------------------------------------------------------------------

    def _sanitize_session_id(self, session_id: str) -> str:
        return session_id.replace("-", "").replace(" ", "_")

    def validate_session_id(self, session_id: str) -> bool:

        if not session_id or not session_id.strip():
            return False

        if ".." in session_id:
            return False

        if "/" in session_id or "\\" in session_id:
            return False

        if len(session_id) > 255:
            return False

        return True

    def get_session_filepath(self, session_id: str) -> Path:
        safe_session_id = self._sanitize_session_id(session_id)
        filename = f"chat_{safe_session_id}.json"
        return CHATS_DATA_DIR / filename

    # ---------------------------------------------------------------------
    # SESSION LOAD
    # ---------------------------------------------------------------------

    def load_session_from_disk(self, session_id: str) -> bool:

        filepath = self.get_session_filepath(session_id)

        if not filepath.exists():
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                chat_dict = json.load(f)

            messages_data = chat_dict.get("messages", [])

            messages = []

            for msg in messages_data:

                role = msg.get("role")
                content = msg.get("content", "")

                if role not in VALID_ROLES:
                    continue

                messages.append(
                    ChatMessage(
                        role=role,
                        content=content
                    )
                )

            self.sessions[session_id] = messages

            logger.info("Loaded session: %s", session_id)

            return True

        except Exception as e:
            logger.warning(
                "Failed to load session %s: %s",
                session_id,
                e
            )
            return False

    # ---------------------------------------------------------------------
    # CREATE / GET SESSION
    # ---------------------------------------------------------------------

    def get_or_create_session(
        self,
        session_id: Optional[str] = None
    ) -> str:

        if not session_id:
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = []
            return session_id

        if not self.validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id: {session_id}"
            )

        if session_id in self.sessions:
            return session_id

        if self.load_session_from_disk(session_id):
            return session_id

        self.sessions[session_id] = []

        return session_id

    # ---------------------------------------------------------------------
    # MESSAGE MANAGEMENT
    # ---------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ) -> None:

        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")

        content = content.strip()

        if not content:
            raise ValueError("Message content cannot be empty.")

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(
            ChatMessage(
                role=role,
                content=content
            )
        )

    def get_chat_history(
        self,
        session_id: str
    ) -> List[ChatMessage]:

        return self.sessions.get(session_id, [])

    # ---------------------------------------------------------------------
    # FORMAT HISTORY FOR LLM
    # ---------------------------------------------------------------------

    def format_history_for_llm(
        self,
        session_id: str,
        exclude_last: bool = False
    ) -> List[Tuple[str, str]]:

        messages = self.get_chat_history(session_id)

        if exclude_last and messages:
            messages = messages[:-1]

        history = []

        i = 0

        while i < len(messages) - 1:

            user_msg = messages[i]
            assistant_msg = messages[i + 1]

            if (
                user_msg.role == "user"
                and assistant_msg.role == "assistant"
            ):
                history.append(
                    (
                        user_msg.content,
                        assistant_msg.content
                    )
                )
                i += 2
            else:
                i += 1

        return history[-MAX_CHAT_HISTORY_TURNS:]

    # ---------------------------------------------------------------------
    # GENERAL CHAT
    # ---------------------------------------------------------------------

    def process_message(
        self,
        session_id: str,
        user_message: str
    ) -> str:

        self.add_message(
            session_id,
            "user",
            user_message
        )

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        response = self.groq_service.get_response(
            question=user_message,
            chat_history=chat_history
        )

        self.add_message(
            session_id,
            "assistant",
            response
        )

        self.save_chat_session(session_id)

        return response

    # ---------------------------------------------------------------------
    # REALTIME CHAT
    # ---------------------------------------------------------------------

    def process_realtime_message(
        self,
        session_id: str,
        user_message: str
    ) -> str:

        if not self.realtime_service:
            raise ValueError(
                "Realtime service is not initialized."
            )

        self.add_message(
            session_id,
            "user",
            user_message
        )

        chat_history = self.format_history_for_llm(
            session_id,
            exclude_last=True
        )

        response = self.realtime_service.get_response(
            question=user_message,
            chat_history=chat_history
        )

        self.add_message(
            session_id,
            "assistant",
            response
        )

        self.save_chat_session(session_id)

        return response

    # ---------------------------------------------------------------------
    # SAVE SESSION
    # ---------------------------------------------------------------------

    def save_chat_session(
        self,
        session_id: str
    ) -> None:

        messages = self.sessions.get(session_id)

        if not messages:
            return

        filepath = self.get_session_filepath(session_id)

        chat_dict = {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
            ]
        }

        try:

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(
                    chat_dict,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            logger.info(
                "Saved session: %s",
                session_id
            )

        except Exception as e:
            logger.error(
                "Failed to save session %s: %s",
                session_id,
                e
            )