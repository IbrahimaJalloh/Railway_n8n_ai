# Utilise Python 3.12 slim
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copie requirements
COPY requirements.txt .

# Installe dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code
COPY . .

# Expose le port (informationnel pour Railway)
EXPOSE 3000

# Lance l'app directement avec Python (gère PORT correctement)
CMD ["python", "-c", "import os, subprocess, sys; subprocess.run([sys.executable, '-m', 'uvicorn', 'app:app', '--host', '0.0.0.0', '--port', os.getenv('PORT', '3000')])"]
