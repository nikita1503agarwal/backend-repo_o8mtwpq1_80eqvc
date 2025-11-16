"""
Database Schemas for BMS Meta (MVP Stage 1)

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercase of the class name. Example: User -> "user" collection.

Collections:
- user
- coach
- bms_meta_card
- extended_anamnesis
- daily_checkins
- scores
- chat_messages
- coach_assignments
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime


class User(BaseModel):
    email: EmailStr
    password_hash: str = Field(..., description="SHA256 hash of the password")
    full_name: str
    onboarded: bool = False
    consent_share_with_coach: bool = False
    role: Literal["user"] = "user"


class Coach(BaseModel):
    email: EmailStr
    password_hash: str
    full_name: str
    specialty: Optional[str] = None
    role: Literal["coach"] = "coach"


class ExtendedAnamnesis(BaseModel):
    user_id: str
    body_goals: Optional[str] = None
    mind_goals: Optional[str] = None
    soul_goals: Optional[str] = None
    habits: Optional[str] = None
    challenges: Optional[str] = None
    preferred_practices: Optional[str] = None


class BmsMetaCard(BaseModel):
    user_id: str
    body_score: float
    mind_score: float
    soul_score: float
    notes: Optional[str] = None


class DailyCheckin(BaseModel):
    user_id: str
    emotional_state: Literal["low", "neutral", "high"]
    energy_level: int = Field(..., ge=1, le=10)
    hydration_goal_met: bool
    micro_action_completed: bool
    sense_of_purpose: int = Field(..., ge=1, le=10)
    reflection_text: Optional[str] = None
    timestamp: Optional[datetime] = None


class Scores(BaseModel):
    user_id: str
    attention_score: float = 50.0
    awareness_score: float = 50.0
    last_checkin_at: Optional[datetime] = None


class ChatMessage(BaseModel):
    conversation_id: str  # user_id:coach_id
    sender_id: str
    receiver_id: str
    sender_role: Literal["user", "coach"]
    message: str
    created_at: Optional[datetime] = None


class CoachAssignment(BaseModel):
    coach_id: str
    user_id: str
    active: bool = True
