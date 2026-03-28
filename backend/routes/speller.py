"""routes/speller.py — Decode EDF dengan EEGNet, SWLDA, dan Hybrid ET."""
import tempfile, os, logging
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session
from db.database import get_db
import numpy as np

logger = logging.getLogger("p300_api.speller")
router = APIRouter()


@router.post("/decode-edf")
async def decode_edf(
    file:        UploadFile = File(...),
    model_id:    str   = Form("latest"),
    alpha_eeg:   float = Form(0.2),
    alpha_et:    float = Form(0.8),
    max_flashes: int   = Form(0),
    db: Session = Depends(get_db),
):
    """
    Decode karakter dari EDF.
    Menghasilkan perbandingan 5 metode:
    - EEGNet only
    - SWLDA only
    - ET only
    - Hybrid EEGNet × ET
    - Hybrid SWLDA × ET
    """
    from services.inference import load_model, decode_session
    from services.preprocessing import parse_edf_sessions, P300_CHANNELS

    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        model_dict = load_model(model_id, db=db)
        d = parse_edf_sessions(tmp_path, P300_CHANNELS)

        results = []
        for sess in d["sessions"]:
            if max_flashes > 0:
                sess = dict(sess)
                sess["onsets"]         = sess["onsets"][:max_flashes]
                sess["stim_types"]     = sess["stim_types"][:max_flashes]
                sess["chars_at_flash"] = sess["chars_at_flash"][:, :max_flashes]

            res = decode_session(
                model_dict, d["preprocessed"], sess,
                et_signals=d["et_signals"],
                has_et=d["has_et"],
                alpha_eeg=alpha_eeg, alpha_et=alpha_et,
                fs=d["fs"],
            )
            results.append(res)

        n = len(results)
        if n == 0:
            return {"filename": file.filename, "model_id": model_id,
                    "has_et": d["has_et"], "n_sessions": 0, "results": []}

        def _acc(key):
            return sum(r.get(key, False) for r in results) / n

        def _text(key):
            return "".join(r.get(key, {}).get("name", "?") for r in results)

        out = {
            "filename":   file.filename,
            "model_id":   model_id,
            "has_et":     d["has_et"],
            "n_sessions": n,
            "accuracy_eegnet": _acc("eegnet_correct"),
            "accuracy_swlda":  _acc("swlda_correct"),
            "decoded_text_eegnet": _text("eegnet_pred"),
            "decoded_text_swlda":  _text("swlda_pred"),
            # backward compat key untuk results_page
            "accuracy_eeg": _acc("eegnet_correct"),
            "decoded_text_eeg": _text("eegnet_pred"),
            "results": results,
        }

        if d["has_et"]:
            out.update({
                "accuracy_et":            _acc("et_correct"),
                "accuracy_hybrid_eegnet": _acc("hybrid_eegnet_correct"),
                "accuracy_hybrid_swlda":  _acc("hybrid_swlda_correct"),
                "decoded_text_et":            _text("et_pred"),
                "decoded_text_hybrid_eegnet": _text("hybrid_eegnet_pred"),
                "decoded_text_hybrid_swlda":  _text("hybrid_swlda_pred"),
                # backward compat
                "accuracy_hybrid":   _acc("hybrid_eegnet_correct"),
                "decoded_text_hybrid": _text("hybrid_eegnet_pred"),
            })

        return out

    finally:
        os.unlink(tmp_path)
