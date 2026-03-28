"""routes/train.py"""
import logging
from typing import List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from db.database import get_db
from services.training import TrainingService

logger = logging.getLogger("p300_api.train")
router = APIRouter()
_svc   = TrainingService()

P300_CH = ["EEG_Cz","EEG_Pz","EEG_P3","EEG_P4","EEG_Oz"]


class TrainRequest(BaseModel):
    session_id:    Optional[str] = None
    data_path:     Optional[str] = None
    channels:      List[str] = P300_CH
    n_epochs:      int   = Field(50, ge=5, le=200)
    n_folds:       int   = Field(5, ge=2, le=10)
    batch_size:    int   = Field(32, ge=8, le=256)
    learning_rate: float = Field(1e-3, gt=0)
    focal_alpha:   float = Field(0.75)
    focal_gamma:   float = Field(2.0)


@router.post("")
async def start_training(req: TrainRequest, bg: BackgroundTasks,
                          db: Session = Depends(get_db)):
    # Resolve data path dari session_id jika ada
    data_path = req.data_path
    if not data_path and req.session_id:
        from db.database import RecordingSession
        sess = db.query(RecordingSession).filter(
            RecordingSession.session_id == req.session_id).first()
        if sess:
            data_path = sess.folder_path
    if not data_path:
        raise HTTPException(422, "Perlu session_id atau data_path")

    config = req.dict()
    config["data_path"] = data_path
    job_id = _svc.create_job(config, db=db)
    bg.add_task(_svc.run_training, job_id, config, db)
    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}")
def get_status(job_id: str, db: Session = Depends(get_db)):
    s = _svc.get_status(job_id, db=db)
    if s is None:
        raise HTTPException(404, f"Job {job_id} tidak ditemukan")
    return s


@router.get("")
def list_jobs(db: Session = Depends(get_db)):
    from db.database import TrainingJob
    rows = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(50).all()
    return {"jobs": [{"job_id": r.job_id, "status": r.status,
                       "model_id": r.model_id, "error": r.error_msg,
                       "created_at": str(r.created_at)} for r in rows]}
