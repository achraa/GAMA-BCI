"""routes/upload.py"""
import logging, os, shutil, uuid, zipfile
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.database import RecordingSession, get_db

logger = logging.getLogger("p300_api.upload")
router = APIRouter()
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)


class PathRequest(BaseModel):
    folder_path: str
    subject_id: Optional[str] = None
    notes: Optional[str] = None


def _scan_meta(folder: Path) -> dict:
    edf_files = list(folder.rglob("*.edf")) + list(folder.rglob("*.EDF"))
    meta = {"n_edf_files": len(edf_files), "sfreq": None,
            "n_channels": None, "has_et": False}
    if edf_files:
        try:
            from services.preprocessing import read_edf
            labels, _, sfreqs = read_edf(str(edf_files[0]))
            meta["sfreq"]     = float(sfreqs[0])
            meta["n_channels"] = len([l for l in labels if l.startswith("EEG")])
            meta["has_et"]    = "ETLeftEyeGazeX" in labels
            meta["channel_names"] = [l for l in labels if l.startswith("EEG")][:20]
        except Exception as e:
            logger.warning("Meta scan error: %s", e)
    return meta


@router.post("/study-folder")
async def upload_zip(
    file: UploadFile = File(...),
    subject_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(415, "Hanya ZIP yang diterima.")
    content = await file.read()
    if len(content) / 1024**2 > 2048:
        raise HTTPException(413, "File terlalu besar (max 2GB).")
    session_id  = str(uuid.uuid4())[:12]
    session_dir = DATA_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    try:
        import io
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            zf.extractall(path=session_dir)
    except zipfile.BadZipFile:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(422, "ZIP rusak atau tidak valid.")
    meta = _scan_meta(session_dir)
    if meta["n_edf_files"] == 0:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise HTTPException(422, "Tidak ada file EDF di dalam ZIP.")
    row = RecordingSession(session_id=session_id, folder_path=str(session_dir),
                            n_edf_files=meta["n_edf_files"], sfreq=meta.get("sfreq"),
                            n_channels=meta.get("n_channels"), has_et=meta.get("has_et", False),
                            status="uploaded", meta=meta)
    db.add(row); db.commit()
    return {"session_id": session_id, **meta,
            "channel_names_preview": meta.get("channel_names", [])[:10]}


@router.post("/from-path")
async def register_path(req: PathRequest, db: Session = Depends(get_db)):
    folder = Path(req.folder_path.strip().strip('"')).expanduser()
    if not folder.is_absolute():
        folder = folder.resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(404, f"Folder tidak ditemukan: {folder}")
    meta = _scan_meta(folder)
    if meta["n_edf_files"] == 0:
        raise HTTPException(422, "Tidak ada EDF di folder tersebut.")
    session_id = str(uuid.uuid4())[:12]
    row = RecordingSession(session_id=session_id, folder_path=str(folder),
                            n_edf_files=meta["n_edf_files"], sfreq=meta.get("sfreq"),
                            n_channels=meta.get("n_channels"), has_et=meta.get("has_et", False),
                            status="uploaded", meta={**meta, "source": "from-path"})
    db.add(row); db.commit()
    return {"session_id": session_id, **meta,
            "channel_names_preview": meta.get("channel_names", [])[:10],
            "folder_path": str(folder)}


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    rows = db.query(RecordingSession).order_by(RecordingSession.uploaded_at.desc()).all()
    return {"sessions": [{"session_id": r.session_id, "n_edf_files": r.n_edf_files,
                           "sfreq": r.sfreq, "has_et": r.has_et, "status": r.status,
                           "uploaded_at": str(r.uploaded_at)} for r in rows]}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    row = db.query(RecordingSession).filter(RecordingSession.session_id == session_id).first()
    if not row:
        raise HTTPException(404, "Session tidak ditemukan")
    db.delete(row); db.commit()
    return {"status": "deleted", "session_id": session_id}
