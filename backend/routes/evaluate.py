"""routes/evaluate.py — Evaluasi model pada EDF file."""
import tempfile, os
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session
from db.database import get_db
import numpy as np

router = APIRouter()


@router.post("/edf")
async def evaluate_edf(
    file: UploadFile = File(...),
    model_id: str = Form("latest"),
    db: Session = Depends(get_db),
):
    from services.inference import load_model, predict_probs_eegnet as predict_probs
    from services.preprocessing import parse_edf_sessions, P300_CHANNELS
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        model_dict = load_model(model_id, db=db)
        d = parse_edf_sessions(tmp_path, P300_CHANNELS)
        X = d["X"]; y = d["y"]
        if len(X) == 0:
            return {"error": "Tidak ada epoch valid"}

        probs  = predict_probs(model_dict, X)
        y_bin  = (probs > 0.3).astype(int)
        auc    = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5
        bal_acc = float(balanced_accuracy_score(y, y_bin))
        f1     = float(f1_score(y, y_bin, zero_division=0))

        return {
            "model_id": model_id,
            "filename": file.filename,
            "n_epochs": len(X),
            "n_target": int(y.sum()),
            "n_nontarget": int((y==0).sum()),
            "metrics": {"roc_auc": auc, "balanced_accuracy": bal_acc,
                        "f1": f1, "imbalance_ratio": f"1:{int((y==0).sum()//max(1,y.sum()))}"},
        }
    finally:
        os.unlink(tmp_path)
