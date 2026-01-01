# app_pro.py - API FastAPI pro sécurisée + monétisée
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from openai import OpenAI
import redis.asyncio as redis
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

# ===== CONFIGURATION =====
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/local.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
MASTER_API_KEY = os.getenv("MASTER_API_KEY", "sk-master-dev-key-change-in-prod")

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== DATABASE =====
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class APIKey(Base):
    """Modèle clé API cliente."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    name = Column(String)
    owner_email = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    requests_count = Column(Integer, default=0)
    is_active = Column(Integer, default=1)

class UsageLog(Base):
    """Modèle log d'usage (pour facturation)."""
    __tablename__ = "usage_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    api_key = Column(String, index=True)
    endpoint = Column(String)
    tokens_used = Column(Integer)
    response_time_ms = Column(Integer)
    cost_cents = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

# Créer tables
Base.metadata.create_all(bind=engine)

# ===== FASTAPI APP =====
app = FastAPI(
    title="Railway n8n AI API Pro",
    description="API sécurisée monétisée pour accès IA",
    version="1.0.0"
)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== RATE LIMITER =====
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# ===== DEPENDENCIES =====
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_api_key(request: Request, db: Session = Depends(get_db)) -> str:
    """Vérifie clé API + log usage."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key requis (Bearer sk-...)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    api_key = auth_header.split(" ")[1]
    
    # Vérify dans DB
    db_key = db.query(APIKey).filter(APIKey.key == api_key, APIKey.is_active == 1).first()
    if not db_key:
        logger.warning(f"API Key invalide: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key invalide ou inactive"
        )
    
    # Update last_used
    db_key.last_used = datetime.utcnow()
    db_key.requests_count += 1
    db.commit()
    
    return api_key

# ===== OPENAI =====
client = OpenAI(api_key=OPENAI_KEY)

# ===== MODELS =====
class ChatRequest(BaseModel):
    message: str
    model_config = ConfigDict(extra='forbid')

class ChatResponse(BaseModel):
    reply: str
    api_key_prefix: str
    tokens_used: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str

class CreateAPIKeyRequest(BaseModel):
    name: str
    owner_email: str

class APIKeyResponse(BaseModel):
    key: str  # Affiché UNE FOIS
    name: str
    created_at: str

# ===== ROUTES =====

@app.get("/health", response_model=HealthResponse)
@limiter.limit("100/minute")
async def health(request: Request):
    """Health check public."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("50/minute")
async def chat(
    req: ChatRequest,
    request: Request,
    db: Session = Depends(get_db),
    api_key: str = Depends(verify_api_key)
):
    """Chat IA sécurisé (authentification + rate-limit)."""
    import time
    start_time = time.time()
    
    try:
        # Appel OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un assistant IA utile."},
                {"role": "user", "content": req.message}
            ],
            max_tokens=500
        )
        
        reply = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        # Log usage pour facturation
        response_time_ms = int((time.time() - start_time) * 1000)
        cost_cents = tokens * 1  # $0.01 per 100 tokens (ajuste selon GPT pricing)
        
        usage = UsageLog(
            api_key=api_key,
            endpoint="/chat",
            tokens_used=tokens,
            response_time_ms=response_time_ms,
            cost_cents=cost_cents
        )
        db.add(usage)
        db.commit()
        
        logger.info(f"Chat: {tokens} tokens, {cost_cents/100:.2f}$ (key: {api_key[:10]}...)")
        
        return ChatResponse(
            reply=reply,
            api_key_prefix=api_key[:10] + "...",
            tokens_used=tokens
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/keys", response_model=APIKeyResponse)
async def create_api_key(
    req: CreateAPIKeyRequest,
    master_key: str,
    db: Session = Depends(get_db)
):
    """Crée une nouvelle clé API (admin only)."""
    if master_key != MASTER_API_KEY:
        raise HTTPException(status_code=403, detail="Master key invalide")
    
    new_key = f"sk-{secrets.token_urlsafe(32)}"
    
    db_key = APIKey(
        key=new_key,
        name=req.name,
        owner_email=req.owner_email
    )
    db.add(db_key)
    db.commit()
    
    logger.info(f"Created key for {req.owner_email}")
    
    return APIKeyResponse(
        key=new_key,  # À copier immédiatement
        name=req.name,
        created_at=datetime.utcnow().isoformat()
    )

@app.get("/admin/usage/{api_key}")
async def get_usage(
    api_key: str,
    master_key: str,
    db: Session = Depends(get_db)
):
    """Récupère usage (facturation) pour une clé."""
    if master_key != MASTER_API_KEY:
        raise HTTPException(status_code=403, detail="Master key invalide")
    
    usage_list = db.query(UsageLog).filter(UsageLog.api_key == api_key).all()
    
    total_tokens = sum(u.tokens_used for u in usage_list)
    total_cost_cents = sum(u.cost_cents for u in usage_list)
    
    return {
        "api_key": api_key[:10] + "...",
        "total_requests": len(usage_list),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost_cents / 100
    }

@app.on_event("startup")
async def startup():
    print("✅ FastAPI Pro API lancée")
    print(f"📍 OPENAI_API_KEY: {('✅' if OPENAI_API_KEY else '❌')}")
    print(f"📍 DATABASE: {DATABASE_URL}")
    print(f"📍 RATE_LIMIT: {RATE_LIMIT_ENABLED}")
    print(f"✨ Endpoints: /docs, /chat, /health, /admin/*")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
