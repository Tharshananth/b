"""
Main Application - Simplified
No Redis, No 5-minute context window
Uses LangChain ConversationBufferWindowMemory
"""
import sys
import io
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
from contextlib import asynccontextmanager

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, get_api_config
from utils.logger import setup_logger
from routers import documents_router, health_router, config_router, feedback_router
from routers.chat import router as chat_router  # Import simplified chat router
from vector_db.retriever import get_retriever
from utils.document_loader import DocumentLoader
from database import init_db

# Setup logging
logger = setup_logger("app")

# Load configuration
config = get_config()
api_config = get_api_config()

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {config.app.name} v{config.app.version}")
    logger.info(f"Environment: {config.app.environment}")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    try:
        # Initialize vector store
        logger.info("Initializing vector store...")
        retriever = get_retriever()
        doc_count = retriever.get_vector_store().get_document_count()
        
        if doc_count == 0:
            logger.info("No documents in vector store, loading from data directory...")
            data_dir = config.documents.data_dir
            docs = DocumentLoader.load_directory(data_dir)
            
            if docs:
                chunks = retriever.get_vector_store().add_documents(docs)
                logger.info(f"Loaded {len(docs)} documents ({chunks} chunks)")
            else:
                logger.warning("No documents found in data directory")
        else:
            logger.info(f"Vector store ready with {doc_count} documents")
        
        # Log available LLM providers
        from llm.factory import get_llm_factory
        factory = get_llm_factory()
        available = factory.get_available_providers()
        logger.info(f"Available LLM providers: {', '.join(available) if available else 'None'}")
        logger.info(f"Default provider: {config.llm.default_provider}")
        
        logger.info("Startup complete")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.warning("Application started with errors")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="VoxelBox RAG Chatbot",
    description="AI-powered chatbot with RAG capabilities",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── ADD MIDDLEWARE IN REVERSE ORDER ────────────────────────────────────────
# Note: app.add_middleware() executes in REVERSE ORDER of addition
# So add CORS LAST so it executes FIRST in the chain

# Add GZip compression (executes 3rd)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add security middleware (executes 2nd)
BLOCKED_PATTERNS = [
    "etc/passwd", "etc/shadow",
    "../", "..\\",
    ".env", ".git",
    "wp-admin", "wp-login",
    "_next/server",
    "phpmyadmin",
    "config.php",
]

@app.middleware("http")
async def block_malicious_requests(request: Request, call_next):
    """Block common attack patterns - skip for CORS preflight"""
    # Skip security checks for CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)
    
    path = str(request.url.path).lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern in path:
            logger.warning(
                f"Blocked malicious request from {request.client.host} → {path}"
            )
            return JSONResponse(
                status_code=403,
                content={"error": "Forbidden"}
            )

    return await call_next(request)

# Add CORS middleware LAST (executes FIRST) ─────────────────────────────────
if api_config.cors.enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors.allow_origins,
        allow_credentials=api_config.cors.allow_credentials,
        allow_methods=api_config.cors.allow_methods,
        allow_headers=api_config.cors.allow_headers,
    )
    logger.info("CORS enabled - allowed origins: %s", api_config.cors.allow_origins)
else:
    logger.warning("CORS is DISABLED in config - frontend will not be able to connect!")

# Register routers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(health_router)
app.include_router(config_router)
app.include_router(feedback_router)

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint"""
    return {
        "name": config.app.name,
        "version": config.app.version,
        "status": "operational",
        "docs": "/docs"
    }

# Health check
@app.get("/health", include_in_schema=False)
async def health():
    """Simple health check"""
    return {"status": "healthy"}

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.app.debug else "An error occurred"
        }
    )

# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    port = api_config.port
    host = api_config.host
    reload = api_config.reload
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload mode: {reload}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.logging.level.lower(),
        access_log=True
    )