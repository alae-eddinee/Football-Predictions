"""
Vercel serverless entry point.
Exposes the FastAPI app as an ASGI handler for Vercel's Python runtime.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.server import app  # noqa: F401 — Vercel picks up `app` as the ASGI handler
