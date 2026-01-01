# 🚀 Railway n8n AI (FastAPI + OpenAI + Postgres + Redis + Rate-Limit)

[![Deploy to Railway](https://railway.app/button.svg)](https://railway.app/new)
[![GitHub Repo stars](https://img.shields.io/github/stars/IbrahimaJalloh/Railway_n8n_ai?style=social)](https://github.com/IbrahimaJalloh/Railway_n8n_ai)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)

**IA Agentic prod-ready** : ChatGPT + n8n workflows + DB persistante + Rate-limit.

![Demo](screenshots/demo.gif) <!-- Ajoutez GIF -->

## 🎯 Quickstart Local

```bash
git clone https://github.com/IbrahimaJalloh/Railway_n8n_ai.git
cd Railway_n8n_ai
pip install -r requirements.txt
cp .env.example .env
# Éditez .env : OPENAI_API_KEY=sk-...
uvicorn app:app --reload --port 3000
