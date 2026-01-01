#!/usr/bin/env python3
import os
import sys
import subprocess

# Récupère PORT
port = os.getenv("PORT", "3000")
print(f"🚀 Démarrage sur le port {port}...")

# Lance uvicorn
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "app:app",
    "--host", "0.0.0.0",
    "--port", port
])
