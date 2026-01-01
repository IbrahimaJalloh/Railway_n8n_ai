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

# Expose le port (informationnel)
EXPOSE 3000

# ✅ OPTION 1 : Avec script shell (RECOMMANDÉE)
# Crée un script entrypoint.sh qui gère la variable PORT correctement
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
CMD ["/entrypoint.sh"]

# ✅ OPTION 2 : Directement avec Python (ALTERNATIVE)
# CMD ["python", "-c", "import os, subprocess, sys; subprocess.run([sys.executable, '-m', 'uvicorn', 'app_pro:app', '--host', '0.0.0.0', '--port', os.getenv('PORT', '3000')])"]
