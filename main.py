import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from hashlib import sha256

from database import db
from schemas import (
    User, Coach, ExtendedAnamnesis, BmsMetaCard, DailyCheckin,
    Scores, ChatMessage, CoachAssignment
)

app = FastAPI(title="BMS Meta API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility helpers

def hash_password(password: str) -> str:
    return sha256(password.encode()).hexdigest()


def simple_reflection_score(text: Optional[str]) -> float:
    if not text:
        return 0.0
    length = min(len(text), 600)
    punctuation_bonus = sum(ch in ".,!?:;" for ch in text) * 2
    return min(100.0, (length / 6.0) + punctuation_bonus)


def decay_attention(attention: float, last_checkin: Optional[datetime]) -> float:
    if not last_checkin:
        return attention
    if isinstance(last_checkin, str):
        try:
            last_checkin = datetime.fromisoformat(last_checkin.replace("Z", "+00:00"))
        except Exception:
            last_checkin = None
    if last_checkin:
        days = (datetime.now(timezone.utc) - last_checkin).days
        if days >= 7:
            attention *= 0.85  # 15% decay
    return max(0.0, min(100.0, attention))


# Auth models

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "user"  # or "coach"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    user_id: str
    role: str
    full_name: str


# Auth endpoints (MVP without JWT)

@app.post("/auth/register", response_model=LoginResponse)
def register(req: RegisterRequest):
    coll_name = "user" if req.role == "user" else "coach"
    coll = db[coll_name]

    existing = coll.find_one({"email": str(req.email)})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    if req.role == "coach":
        model = Coach(email=req.email, password_hash=hash_password(req.password), full_name=req.full_name)
        _id = coll.insert_one({**model.model_dump(), "created_at": datetime.now(timezone.utc)}).inserted_id
        return LoginResponse(user_id=str(_id), role="coach", full_name=req.full_name)
    else:
        model = User(email=req.email, password_hash=hash_password(req.password), full_name=req.full_name)
        _id = coll.insert_one({**model.model_dump(), "created_at": datetime.now(timezone.utc)}).inserted_id
        # initialize scores
        score = Scores(user_id=str(_id))
        db["scores"].insert_one({**score.model_dump(), "created_at": datetime.now(timezone.utc)})
        return LoginResponse(user_id=str(_id), role="user", full_name=req.full_name)


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    for collection_name in ["user", "coach"]:
        acc = db[collection_name].find_one({"email": str(req.email)})
        if acc and acc.get("password_hash") == hash_password(req.password):
            return LoginResponse(user_id=str(acc.get("_id")), role="coach" if collection_name == "coach" else "user", full_name=acc.get("full_name", ""))
    raise HTTPException(status_code=401, detail="Invalid credentials")


# Onboarding: Extended Anamnesis -> Meta Card + Initial Scores

class OnboardingRequest(BaseModel):
    user_id: str
    consent_share_with_coach: bool
    body_goals: Optional[str] = None
    mind_goals: Optional[str] = None
    soul_goals: Optional[str] = None
    habits: Optional[str] = None
    challenges: Optional[str] = None
    preferred_practices: Optional[str] = None


@app.post("/onboarding/submit")
def submit_onboarding(req: OnboardingRequest):
    from bson import ObjectId
    try:
        oid = ObjectId(req.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    user = db["user"].find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db["user"].update_one({"_id": oid}, {"$set": {"onboarded": True, "consent_share_with_coach": req.consent_share_with_coach}})

    anam = ExtendedAnamnesis(
        user_id=req.user_id,
        body_goals=req.body_goals,
        mind_goals=req.mind_goals,
        soul_goals=req.soul_goals,
        habits=req.habits,
        challenges=req.challenges,
        preferred_practices=req.preferred_practices,
    )
    db["extendedanamnesis"].insert_one({**anam.model_dump(), "created_at": datetime.now(timezone.utc)})

    # Initial BMS scores (naive based on presence/length of goals)
    def base_score(text: Optional[str]) -> float:
        if not text:
            return 50.0
        l = min(len(text), 500)
        return float(min(90.0, 50.0 + l / 10.0))

    bms = BmsMetaCard(
        user_id=req.user_id,
        body_score=base_score(req.body_goals),
        mind_score=base_score(req.mind_goals),
        soul_score=base_score(req.soul_goals),
        notes=None,
    )
    db["bmsmetacard"].insert_one({**bms.model_dump(), "created_at": datetime.now(timezone.utc)})

    # Update scores
    scores = db["scores"].find_one({"user_id": req.user_id})
    if not scores:
        scores = Scores(user_id=req.user_id).model_dump()
    scores["awareness_score"] = float(simple_reflection_score(" ".join(filter(None, [req.body_goals, req.mind_goals, req.soul_goals, req.habits, req.challenges]))))
    scores["attention_score"] = 60.0
    db["scores"].update_one({"user_id": req.user_id}, {"$set": scores}, upsert=True)

    return {"status": "ok"}


# Daily Check-in

class DailyCheckinRequest(BaseModel):
    user_id: str
    emotional_state: str
    energy_level: int
    hydration_goal_met: bool
    micro_action_completed: bool
    sense_of_purpose: int
    reflection_text: Optional[str] = None


@app.post("/checkin/submit")
def submit_checkin(req: DailyCheckinRequest):
    from bson import ObjectId
    try:
        oid = ObjectId(req.user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    user = db["user"].find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    checkin = DailyCheckin(
        user_id=req.user_id,
        emotional_state=req.emotional_state,
        energy_level=req.energy_level,
        hydration_goal_met=req.hydration_goal_met,
        micro_action_completed=req.micro_action_completed,
        sense_of_purpose=req.sense_of_purpose,
        reflection_text=req.reflection_text,
        timestamp=datetime.now(timezone.utc),
    )
    db["dailycheckin"].insert_one({**checkin.model_dump(), "created_at": datetime.now(timezone.utc)})

    # Update scores
    scores = db["scores"].find_one({"user_id": req.user_id}) or Scores(user_id=req.user_id).model_dump()

    attention = float(scores.get("attention_score", 50.0))
    awareness = float(scores.get("awareness_score", 50.0))

    # Attention: base + consistency + habits
    attention += 2.0  # for checking in today
    if req.hydration_goal_met:
        attention += 1.0
    if req.micro_action_completed:
        attention += 1.5
    attention = decay_attention(attention, scores.get("last_checkin_at"))

    # Awareness: combine reflection depth and self-ratings
    awareness_delta = 0.3 * req.sense_of_purpose + 0.2 * (req.energy_level)
    awareness_text = 0.5 * (simple_reflection_score(req.reflection_text) / 10.0)
    awareness += awareness_delta + awareness_text

    scores.update({
        "attention_score": float(max(0.0, min(100.0, attention))),
        "awareness_score": float(max(0.0, min(100.0, awareness))),
        "last_checkin_at": datetime.now(timezone.utc),
    })

    db["scores"].update_one({"user_id": req.user_id}, {"$set": scores}, upsert=True)

    return {"status": "ok"}


# Dashboard endpoints

@app.get("/dashboard/{user_id}")
def get_dashboard(user_id: str):
    from bson import ObjectId
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    user = db["user"].find_one({"_id": oid})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    bms = db["bmsmetacard"].find_one({"user_id": user_id}, sort=[("created_at", -1)])
    scores = db["scores"].find_one({"user_id": user_id})
    last_checkins = list(db["dailycheckin"].find({"user_id": user_id}).sort("timestamp", -1).limit(14))

    summary = {
        "trend_energy_avg": sum(c.get("energy_level", 0) for c in last_checkins) / max(1, len(last_checkins)),
        "checkin_count": len(last_checkins),
        "hydration_rate": sum(1 for c in last_checkins if c.get("hydration_goal_met")) / max(1, len(last_checkins)),
    }

    return {
        "user": {
            "full_name": user.get("full_name"),
            "onboarded": user.get("onboarded", False),
        },
        "bms": bms,
        "scores": scores,
        "recent_checkins": last_checkins,
        "weekly_summary": summary,
    }


# Coach features

@app.get("/coach/{coach_id}/clients")
def coach_clients(coach_id: str):
    from bson import ObjectId
    assignments = list(db["coachassignment"].find({"coach_id": coach_id, "active": True}))
    user_ids = [a.get("user_id") for a in assignments]
    ids = []
    for uid in user_ids:
        try:
            ids.append(ObjectId(uid))
        except Exception:
            continue
    users = list(db["user"].find({"_id": {"$in": ids}})) if ids else []

    visible = []
    for u in users:
        if not u.get("consent_share_with_coach"):
            continue
        uid = str(u.get("_id"))
        sc = db["scores"].find_one({"user_id": uid}) or {}
        last_check = sc.get("last_checkin_at")
        visible.append({
            "user_id": uid,
            "full_name": u.get("full_name"),
            "attention_score": sc.get("attention_score"),
            "awareness_score": sc.get("awareness_score"),
            "last_activity": last_check,
        })

    return {"clients": visible}


@app.get("/coach/client/{user_id}")
def coach_view_client(user_id: str):
    from bson import ObjectId
    try:
        oid = ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    u = db["user"].find_one({"_id": oid})
    if not u or not u.get("consent_share_with_coach"):
        raise HTTPException(status_code=403, detail="No access to this client")
    bms = db["bmsmetacard"].find_one({"user_id": user_id}, sort=[("created_at", -1)])
    scores = db["scores"].find_one({"user_id": user_id})
    last_checkins = list(db["dailycheckin"].find({"user_id": user_id}).sort("timestamp", -1).limit(30))
    return {"profile": u, "bms": bms, "scores": scores, "checkins": last_checkins}


# Chat system

class SendMessageRequest(BaseModel):
    conversation_id: str  # f"{user_id}:{coach_id}"
    sender_id: str
    receiver_id: str
    sender_role: str  # "user" | "coach"
    message: str


@app.post("/chat/send")
def send_message(req: SendMessageRequest):
    msg = ChatMessage(
        conversation_id=req.conversation_id,
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        sender_role=req.sender_role,
        message=req.message,
        created_at=datetime.now(timezone.utc),
    )
    db["chatmessage"].insert_one({**msg.model_dump(), "created_at": datetime.now(timezone.utc)})

    # Only update awareness score for USER reflections
    if req.sender_role == "user":
        uid = req.sender_id
        sc = db["scores"].find_one({"user_id": uid}) or Scores(user_id=uid).model_dump()
        sc["awareness_score"] = float(max(0.0, min(100.0, sc.get("awareness_score", 50.0) + simple_reflection_score(req.message) / 20.0)))
        db["scores"].update_one({"user_id": uid}, {"$set": sc}, upsert=True)

    return {"status": "sent"}


@app.get("/chat/{conversation_id}")
def get_conversation(conversation_id: str):
    msgs = list(db["chatmessage"].find({"conversation_id": conversation_id}).sort("created_at", 1))
    return {"messages": msgs}


# Health check and DB test

@app.get("/")
def read_root():
    return {"message": "BMS Meta API running"}


@app.get("/test")
def test_database():
    resp = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            resp["database"] = "✅ Connected & Working"
            resp["connection_status"] = "Connected"
            resp["collections"] = db.list_collection_names()
    except Exception as e:
        resp["database"] = f"⚠️  Connected but Error: {str(e)[:60]}"
    return resp


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
