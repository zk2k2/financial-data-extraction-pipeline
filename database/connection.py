import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.invoice import Base

# Use SQLite - creates a file in your workspace
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./invoice_database.db"
)

# SQLite specific engine configuration
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Needed for SQLite with FastAPI
    echo=False  # Set to True for SQL query logging during development
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all tables defined in models"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency function for FastAPI to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_engine():
    """Get the SQLAlchemy engine (useful for Alembic)"""
    return engine