import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from openai import OpenAI
import dotenv

dotenv.load_dotenv()
app = FastAPI(title="IA Railway App")

def get_secret(name: str) -> str:
    val = os.getenv(name)
    if not val or 'dummy' in val.lower():
        print(f"🔒 LOCAL TEST: {name}")
        return "dummy"
    print(f"🔐 PROD: {name} OK")
    return val

openai_key = get_secret("OPENAI_API_KEY")
db_url = get_secret("DATABASE_URL")
client = OpenAI(api_key=openai_key)
engine = create_engine(db_url or "sqlite:///test.db")

REDIS_URL = os.getenv("REDIS_URL")

@app.on_event("startup")
async def startup():
    if REDIS_URL:
        r = await redis.from_url(REDIS_URL)
        await FastAPILimiter.init(r)

@app.get("/")
async def root():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
        return {"status": "OK", "db": result.scalar()}
    except:
        return {"status": "OK local"}

@app.post("/chat", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def chat(body: dict):
    msg = body.get("message", "")[:1000]
    if not msg:
        raise HTTPException(400, "Message?")
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": msg}])
    return {"reply": resp.choices[0].message.content}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
