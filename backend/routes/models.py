"""routes/models.py"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from db.database import get_db, RegisteredModel

router = APIRouter()


@router.get("")
def list_models(db: Session = Depends(get_db)):
    models = db.query(RegisteredModel).order_by(
        RegisteredModel.registered_at.desc()).all()
    return {"models": [{
        "model_id":      m.model_id,
        "is_active":     m.is_active,
        "registered_at": str(m.registered_at),
        "job_id":        m.job_id,
        "avg_bal_acc":   m.avg_bal_acc,
        "avg_roc_auc":   m.avg_roc_auc,
        "avg_f1":        m.avg_f1,
        "avg_recall":    m.avg_recall,
        "n_channels":    m.n_channels,
        "channels_used": m.channels_used,
        "best_threshold": m.best_threshold,
    } for m in models]}


@router.post("/{model_id}/activate")
def activate(model_id: str, db: Session = Depends(get_db)):
    db.query(RegisteredModel).update({"is_active": False})
    m = db.query(RegisteredModel).filter(RegisteredModel.model_id == model_id).first()
    if not m:
        raise HTTPException(404, "Model tidak ditemukan")
    m.is_active = True; db.commit()
    return {"status": "activated", "model_id": model_id}


@router.delete("/{model_id}")
def delete_model(model_id: str, db: Session = Depends(get_db)):
    m = db.query(RegisteredModel).filter(RegisteredModel.model_id == model_id).first()
    if not m:
        raise HTTPException(404, "Model tidak ditemukan")
    db.delete(m); db.commit()
    return {"status": "deleted"}
