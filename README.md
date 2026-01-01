# Railway App IA (FastAPI + OpenAI + Postgres + Redis Rate-Limit)

## Local
cp .env.example .env  # Dummies only
uvicorn app:app --reload

## Railway
1. Vars: OPENAI_API_KEY=sk-vraie, DATABASE_URL=${{Postgres.DATABASE_URL}}
2. New: Postgres + Redis
3. Auto-deploy GitHub push
