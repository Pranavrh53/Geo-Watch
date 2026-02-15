"""
Database models and setup
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

# Database file location
DB_PATH = Path(__file__).parent.parent / "data" / "geowatch.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class CachedTile(Base):
    """Cached satellite tile"""
    __tablename__ = "cached_tiles"
    
    id = Column(Integer, primary_key=True, index=True)
    bbox_hash = Column(String, index=True, nullable=False)  # Hash of bbox coordinates
    date = Column(String, index=True, nullable=False)  # YYYY-MM-DD
    image_path = Column(String, nullable=False)  # Path to cached image
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)  # Cache expiration
    bbox_west = Column(Float)
    bbox_south = Column(Float)
    bbox_east = Column(Float)
    bbox_north = Column(Float)


class AnalysisHistory(Base):
    """User analysis history"""
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    region_name = Column(String)  # Optional name user gives
    bbox_west = Column(Float, nullable=False)
    bbox_south = Column(Float, nullable=False)
    bbox_east = Column(Float, nullable=False)
    bbox_north = Column(Float, nullable=False)
    before_date = Column(String, nullable=False)
    after_date = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="completed")  # pending, processing, completed, failed
    result_json = Column(Text)  # JSON string with analysis results


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized")


if __name__ == "__main__":
    init_db()
