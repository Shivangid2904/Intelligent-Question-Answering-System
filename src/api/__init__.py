"""
IntelliAsk Research — API Package

Exposes the FastAPI application factory, default ASGI application instance,
and configuration dataclass.
"""

from src.api.config import APIConfig
from src.api.main import app, create_app

__all__ = ["app", "create_app", "APIConfig"]
