"""Preview-tool wrapper — sets CWD then starts the catan FastAPI server."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import uvicorn
# Port 8000 is taken by a long-running polymarket-bot on 0.0.0.0, so use 8765
# to avoid the collision. Keep this in sync with launch.json and vite.config.ts.
uvicorn.run("main:app", host="127.0.0.1", port=8765)
