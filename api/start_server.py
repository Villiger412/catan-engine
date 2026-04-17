"""Preview-tool wrapper — sets CWD then starts the catan FastAPI server."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import uvicorn
uvicorn.run("main:app", host="127.0.0.1", port=8000)
