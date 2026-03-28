"""services/inference.py — Inference EEGNet + ET hybrid."""
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("p300_api.inference")
N_CHARS = 72


def load_model(model_id: str, db=None):
    """Load EEGNet model — returns dict."""
    import torch
    from model.eegnet import EEGNet

    if db:
        from db.database import RegisteredModel
        if model_id == "latest":
            rm = db.query(RegisteredModel).filter(
                RegisteredModel.is_active == True).first()
        else:
            rm = db.query(RegisteredModel).filter(
                RegisteredModel.model_id == model_id).first()
        if rm is None:
            raise ValueError(f"Model tidak ditemukan: {model_id}")
        art_dir    = Path(rm.artifact_dir)
        n_channels = rm.n_channels or 5
        n_times    = rm.n_times or 204
    else:
        art_dir = Path("./models") / model_id
        with open(art_dir / "config.json") as f:
            cfg = json.load(f)
        n_channels = cfg.get("n_channels", 5)
        n_times    = cfg.get("n_times", 204)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eegnet_path = art_dir / "eegnet.pt"
    model_path  = art_dir / "model.pt"
    if eegnet_path.exists():
        pt_path = eegnet_path
    elif model_path.exists():
        pt_path = model_path
    else:
        raise FileNotFoundError(f"Tidak ada file model .pt di: {art_dir}")

    model = EEGNet(n_channels=n_channels, n_timepoints=n_times).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    mu  = np.load(art_dir / "mu.npy")
    std = np.load(art_dir / "std.npy")

    return {"eegnet": model, "eegnet_mu": mu, "eegnet_std": std, "device": device}


def predict_probs_eegnet(model_dict, X: np.ndarray) -> np.ndarray:
    import torch
    model  = model_dict["eegnet"]
    mu     = model_dict["eegnet_mu"]
    std    = model_dict["eegnet_std"]
    device = model_dict["device"]

    X_n = (X - mu) / (std + 1e-8)
    X_t = torch.FloatTensor(X_n[:, np.newaxis]).to(device)
    with torch.no_grad():
        raw = model(X_t).cpu().numpy().flatten()
    
    print(f"[DEBUG] n={len(raw)} min={raw.min():.4f} max={raw.max():.4f} mean={raw.mean():.4f} std={raw.std():.4f}")
    return raw

# Alias backward compat
predict_probs = predict_probs_eegnet


def accumulate_probs(probs, valid_mask, chars_at_flash, n_chars=N_CHARS):
    log_p     = np.zeros(n_chars)
    valid_idx = np.where(valid_mask)[0]
    for k, vi in enumerate(valid_idx):
        if vi >= chars_at_flash.shape[1] or k >= len(probs):
            continue
        char_mask = chars_at_flash[:, vi] > 0.5
        p_k = float(np.clip(probs[k], 1e-7, 1 - 1e-7))
        log_p[char_mask]  += np.log(p_k)
        log_p[~char_mask] += np.log(1 - p_k) * 0.1
    lp = log_p - log_p.max()
    return np.exp(lp) / np.exp(lp).sum()


def fuse_eeg_et(p_eeg, p_et, alpha_eeg=0.2, alpha_et=0.8):
    p_eeg = np.clip(p_eeg, 1e-10, 1)
    p_et  = np.clip(p_et,  1e-10, 1)
    log_p = alpha_eeg * np.log(p_eeg) + alpha_et * np.log(p_et)
    log_p -= log_p.max()
    p = np.exp(log_p) / np.exp(log_p).sum()
    return p, int(np.argmax(p))


def decode_session(model_dict, preprocessed, session,
                    et_signals=None, has_et=False,
                    alpha_eeg=0.2, alpha_et=0.8, fs=256.0):
    from services.preprocessing import extract_epochs, compute_et_probability

    onsets      = session["onsets"]
    stim_types  = session["stim_types"]
    chars_flash = session["chars_at_flash"]
    char_info   = session["char_info"]
    char_rows   = np.array([c["row"] for c in char_info])
    char_cols   = np.array([c["col"] for c in char_info])
    target      = session["target"]

    X_sess, _, valid = extract_epochs(preprocessed, onsets, stim_types, fs=fs)

    p_eeg = np.ones(N_CHARS) / N_CHARS
    if len(X_sess) > 0:
        probs = predict_probs_eegnet(model_dict, X_sess)
        p_eeg = accumulate_probs(probs, valid, chars_flash)

    eeg_idx  = int(np.argmax(p_eeg))
    eeg_pred = char_info[eeg_idx]
    eeg_ok   = (eeg_pred["row"] == target["row"] and eeg_pred["col"] == target["col"])

    result = {
        "session_id":        session["session_id"],
        "target":            target,
        "eegnet_pred":       eeg_pred,
        "eegnet_correct":    eeg_ok,
        "eegnet_confidence": float(p_eeg[eeg_idx]),
        "p_eegnet":          p_eeg.tolist(),
        "eeg_pred":          eeg_pred,
        "eeg_correct":       eeg_ok,
        "eeg_confidence":    float(p_eeg[eeg_idx]),
        "p_eeg":             p_eeg.tolist(),
        "has_et":            has_et,
        "n_valid_flash":     int(valid.sum()),
    }

    if has_et and et_signals is not None:
        p_et = compute_et_probability(
            et_signals, onsets, char_rows, char_cols,
            fs=fs, pup_baseline=session.get("pup_baseline", 4.5))
        et_idx  = int(np.argmax(p_et))
        et_pred = char_info[et_idx]
        et_ok   = (et_pred["row"] == target["row"] and et_pred["col"] == target["col"])

        p_hyb, hyb_idx = fuse_eeg_et(p_eeg, p_et, alpha_eeg, alpha_et)
        hyb_pred = char_info[hyb_idx]
        hyb_ok   = (hyb_pred["row"] == target["row"] and hyb_pred["col"] == target["col"])

        result.update({
            "et_pred":               et_pred,
            "et_correct":            et_ok,
            "et_confidence":         float(p_et[et_idx]),
            "hybrid_pred":           hyb_pred,
            "hybrid_correct":        hyb_ok,
            "hybrid_confidence":     float(p_hyb[hyb_idx]),
            "hybrid_eegnet_pred":    hyb_pred,
            "hybrid_eegnet_correct": hyb_ok,
            "hybrid_eegnet_confidence": float(p_hyb[hyb_idx]),
            "p_et":                  p_et.tolist(),
            "p_hybrid":              p_hyb.tolist(),
            "p_hybrid_eegnet":       p_hyb.tolist(),
        })

    return result
