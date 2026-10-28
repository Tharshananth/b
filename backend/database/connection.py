"""Database connection management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pathlib import Path
import logging
from .models import Base

logger = logging.getLogger(__name__)

# Database file path
DB_DIR = Path("data/database")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "feedback.db"

# Create engine
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False}  # Important for SQLite with FastAPI
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"✅ Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise

def get_db():
    """Get database session - FastAPI Dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()