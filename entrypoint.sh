#!/bin/sh
# entrypoint.sh

# Récupère PORT de l'environnement, ou utilise 3000 par défaut
PORT=${PORT:-3000}

# Affiche pour le debug
echo "🚀 Démarrage de l'application sur le port $PORT..."

# Lance uvicorn
exec uvicorn app:app --host 0.0.0.0 --port "$PORT"
