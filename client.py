#!/usr/bin/env python3
"""
🤖 N8N Client Localhost ULTIME - Docker + Tests
Usage: python client.py --help
"""

import requests
import typer
from dotenv import load_dotenv
import os
import json
import subprocess
import time
from rich.console import Console
from rich.table import Table
from rich import print as rprint

load_dotenv()
console = Console()

app = typer.Typer()

class N8NClient:
    def __init__(self, local=False):
        self.base_url = "http://localhost:5678" if local else os.getenv("N8N_WEBHOOK", "https://n8n-production-d333.up.railway.app")
        self.timeout = 120 if local else 60
        self.auth = os.getenv("N8N_BASIC_AUTH", "")  # user:pass si activé
        
    def start_docker(self):
        """Docker n8n local"""
        try:
            subprocess.run(["docker", "ps", "--filter", "name=n8n-local"], capture_output=True)
            rprint("[green]✅ n8n docker running")
        except:
            rprint("[yellow]🚀 Starting n8n docker...")
            cmd = [
                "docker", "run", "-d", "--name", "n8n-local",
                "-p", "5678:5678",
                "-v", "n8n_data:/home/node/.n8n",
                "n8nio/n8n"
            ]
            subprocess.run(cmd)
            time.sleep(10)
            rprint("[green]✅ n8n ready localhost:5678")

    def send_task(self, webhook_path: str, task: str):
        url = f"{self.base_url}/{webhook_path}"
        payload = {"task": task}
        headers = {}
        if self.auth:
            from requests.auth import HTTPBasicAuth
            return requests.post(url, json=payload, auth=HTTPBasicAuth(*self.auth.split(':')), timeout=self.timeout)
        
        r = requests.post(url, json=payload, timeout=self.timeout)
        return r

@app.command()
def local():
    """Start n8n docker + test"""
    client = N8NClient(local=True)
    client.start_docker()
    rprint("[bold blue]🧪 Test localhost[/]")
    r.send_task("webhook-test/106bbca1-f10e-456f-b58f-dd645f07f1c4", "Hello local")

@app.command()
def run(task: str):
    """Single task"""
    client = N8NClient()
    webhook = os.getenv("N8N_WEBHOOK_PATH", "webhook-test/106bbca1-f10e-456f-b58f-dd645f07f1c4")
    r = client.send_task(webhook, task)
    table = Table()
    table.add_column("Status")
    table.add_column("Response")
    table.add_row(str(r.status_code), str(r.json()[:100]))
    console.print(table)

@app.command()
def batch(file: str = "tasks.txt"):
    """Batch from file"""
    client = N8NClient()
    webhook = os.getenv("N8N_WEBHOOK_PATH", "webhook-test/106bbca1-f10e-456f-b58f-dd645f07f1c4")
    with open(file) as f:
        tasks = [line.strip() for line in f if line.strip()]
    for task in tasks:
        r = client.send_task(webhook, task)
        rprint(f"[green]✅ {task} -> {r.status_code}")
        time.sleep(2)

if __name__ == "__main__":
    app()
