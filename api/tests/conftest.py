import sys
from pathlib import Path

# Make sure api/ and the repo root are on the path so the test client can
# import main.py and find catan_engine.
ROOT = Path(__file__).parent.parent.parent
API  = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(API))

import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
