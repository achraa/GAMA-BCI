"""db/database.py — SQLite database."""
import os
from datetime import datetime
from pathlib import Path
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer,
                         JSON, String, Text, Boolean, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

DEFAULT_DB = Path(__file__).resolve().parents[1] / "p300_bci.db"
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB.as_posix()}")

def _pragmas(conn, _):
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()

_args = {"check_same_thread": False} if DB_URL.startswith("sqlite") else {}
engine = create_engine(DB_URL, connect_args=_args,
                        poolclass=StaticPool if ":memory:" in DB_URL else None)
if DB_URL.startswith("sqlite"):
    event.listen(engine, "connect", _pragmas)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    id          = Column(Integer, primary_key=True)
    session_id  = Column(String(64), unique=True, nullable=False, index=True)
    folder_path = Column(Text, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    n_edf_files = Column(Integer, default=0)
    sfreq       = Column(Float, nullable=True)
    n_channels  = Column(Integer, nullable=True)
    has_et      = Column(Boolean, default=False)
    status      = Column(String(32), default="uploaded")
    meta        = Column(JSON, nullable=True)
    jobs        = relationship("TrainingJob", back_populates="session",
                               cascade="all, delete-orphan")

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id          = Column(Integer, primary_key=True)
    job_id      = Column(String(64), unique=True, nullable=False, index=True)
    session_fk  = Column(Integer, ForeignKey("recording_sessions.id"), nullable=True)
    status      = Column(String(32), default="queued")
    config      = Column(JSON, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    started_at  = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_msg   = Column(Text, nullable=True)
    model_id    = Column(String(128), nullable=True)
    logs        = Column(Text, nullable=True)
    session     = relationship("RecordingSession", back_populates="jobs")
    metrics     = relationship("JobMetric", back_populates="job",
                               cascade="all, delete-orphan")

class JobMetric(Base):
    __tablename__ = "job_metrics"
    id                = Column(Integer, primary_key=True)
    job_fk            = Column(Integer, ForeignKey("training_jobs.id"))
    fold              = Column(Integer, nullable=True)
    split             = Column(String(16), nullable=True)
    balanced_accuracy = Column(Float, nullable=True)
    roc_auc           = Column(Float, nullable=True)
    f1                = Column(Float, nullable=True)
    recall            = Column(Float, nullable=True)
    job = relationship("TrainingJob", back_populates="metrics")

class RegisteredModel(Base):
    __tablename__ = "registered_models"
    id            = Column(Integer, primary_key=True)
    model_id      = Column(String(128), unique=True, nullable=False, index=True)
    artifact_dir  = Column(Text)
    is_active     = Column(Boolean, default=False)
    registered_at = Column(DateTime, default=datetime.utcnow)
    job_id        = Column(String(64), nullable=True)
    avg_bal_acc   = Column(Float, nullable=True)
    avg_roc_auc   = Column(Float, nullable=True)
    avg_f1        = Column(Float, nullable=True)
    avg_recall    = Column(Float, nullable=True)
    n_channels    = Column(Integer, nullable=True)
    n_times       = Column(Integer, nullable=True)
    channels_used = Column(JSON, nullable=True)
    best_threshold = Column(Float, default=0.5)
    meta          = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
