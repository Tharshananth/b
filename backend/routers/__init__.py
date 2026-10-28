"""API Routers Package"""
from .chat import router as chat_router
from .config import router as config_router
from .feedback import router as feedback_router

try:
    from .health import router as health_router
except ImportError:
    health_router = None

try:
    from .documents import router as documents_router
except ImportError:
    documents_router = None

__all__ = [
    'chat_router', 
    'documents_router', 
    'health_router', 
    'config_router',
    'feedback_router'
]