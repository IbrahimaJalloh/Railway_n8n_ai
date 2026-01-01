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

# Expose le port par défaut
EXPOSE 3000

# Lance l'app avec la variable PORT injectée par Railway
CMD ["sh", "-c", "uvicorn app_pro:app --host 0.0.0.0 --port ${PORT:-3000}"]
