# tests/conftest.py
import sys
from pathlib import Path

# Add the project root (parent of "tests/") to sys.path so "import src" works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))