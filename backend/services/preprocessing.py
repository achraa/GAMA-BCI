"""services/preprocessing.py — EEG preprocessing + ET features untuk bigP3BCI SE001."""
import logging
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("p300_api.preprocess")

# Semua 16 channel EEG dari dataset bigP3BCI
P300_CHANNELS = [
    "EEG_F3", "EEG_Fz", "EEG_F4", "EEG_T7", "EEG_C3",
    "EEG_Cz", "EEG_C4", "EEG_T8", "EEG_CP3", "EEG_CP4",
    "EEG_P3", "EEG_Pz", "EEG_P4", "EEG_PO7", "EEG_PO8",
    "EEG_Oz"
]
# Channel noise/artifact otot — frontal & temporal
NOISE_CHANNELS = ["EEG_F3", "EEG_Fz", "EEG_F4", "EEG_T7", "EEG_T8"]

FS        = 256.0
EPOCH_END = 0.8
N_SAMPLES = int(EPOCH_END * FS)  # 204 samples

# ET kalibrasi (fitted dari ground-truth data 85 sessions SE001)
CALIB_COL_A = 7.460;  CALIB_COL_B = 0.756
CALIB_ROW_A = 9.839;  CALIB_ROW_B = 0.324
SCREEN_DIST_MM = 700.0
SIGMA_ROW = 0.9;  SIGMA_COL = 1.0

GRID_ROWS = 9;  GRID_COLS = 8
N_CHARS   = GRID_ROWS * GRID_COLS  # 72


# ── EDF Reader ─────────────────────────────────────────────────

def read_edf(filepath: str):
    with open(filepath, "rb") as f:
        f.read(8); f.read(80); f.read(80); f.read(8); f.read(8)
        int(f.read(8).decode("ascii","replace").strip()); f.read(44)
        nr  = int(f.read(8).decode("ascii","replace").strip())
        dur = float(f.read(8).decode("ascii","replace").strip())
        ns  = int(f.read(4).decode("ascii","replace").strip())
        labels   = [f.read(16).decode("ascii","replace").strip() for _ in range(ns)]
        [f.read(80) for _ in range(ns)]; [f.read(8) for _ in range(ns)]
        pmin = [float(f.read(8).decode("ascii","replace").strip()) for _ in range(ns)]
        pmax = [float(f.read(8).decode("ascii","replace").strip()) for _ in range(ns)]
        dmin = [int(f.read(8).decode("ascii","replace").strip()) for _ in range(ns)]
        dmax = [int(f.read(8).decode("ascii","replace").strip()) for _ in range(ns)]
        [f.read(80) for _ in range(ns)]
        nsamp = [int(f.read(8).decode("ascii","replace").strip()) for _ in range(ns)]
        f.read(32 * ns)
        sfreqs = [nsamp[i]/dur for i in range(ns)]
        sigs = [[] for _ in range(ns)]
        for _ in range(nr):
            for i in range(ns):
                if labels[i] == "EDF Annotations":
                    f.read(nsamp[i]*2); sigs[i].extend([0]*nsamp[i])
                else:
                    raw  = f.read(nsamp[i]*2)
                    vals = struct.unpack(f"<{nsamp[i]}h", raw)
                    g    = (pmax[i]-pmin[i])/(dmax[i]-dmin[i]) if dmax[i]!=dmin[i] else 1.0
                    off  = pmin[i] - dmin[i]*g
                    sigs[i].extend([v*g+off for v in vals])
        return labels, [np.array(s) for s in sigs], sfreqs


# ── EEG Preprocessing ──────────────────────────────────────────

def preprocess_eeg(raw_signals, channel_indices, fs=256.0):
    from scipy.signal import butter, filtfilt, iirnotch
    data = np.array([raw_signals[i] for i in channel_indices])
    nyq  = fs / 2
    # Bandpass 0.1–20Hz (lebih ketat, kurangi noise otot)
    b, a = butter(4, [0.1/nyq, 20.0/nyq], btype="band")
    for c in range(data.shape[0]):
        data[c] = filtfilt(b, a, data[c])
    # Notch 50Hz
    b, a = iirnotch(50.0/nyq, 30.0)
    for c in range(data.shape[0]):
        data[c] = filtfilt(b, a, data[c])
    # CAR
    data -= data.mean(axis=0, keepdims=True)
    return data


def extract_epochs(preprocessed, onsets, stim_types, fs=256.0, threshold=100.0):
    s1 = int(EPOCH_END * fs)
    T  = preprocessed.shape[1]
    X_list, y_list, valid_mask = [], [], []
    for i, onset in enumerate(onsets):
        if onset+s1 > T or onset < 0:
            valid_mask.append(False); continue
        epoch = preprocessed[:, onset:onset+s1].copy()
        # Baseline correction 200ms pre-stimulus
        bl_samples = int(0.2 * fs)
        bl_start   = max(0, onset - bl_samples)
        if bl_start < onset:
            bl = preprocessed[:, bl_start:onset].mean(axis=1, keepdims=True)
        else:
            bl = epoch[:, :int(0.05*fs)].mean(axis=1, keepdims=True)
        epoch -= bl
        if np.abs(epoch).max() > threshold:
            valid_mask.append(False); continue
        X_list.append(epoch)
        y_list.append(int(round(stim_types[i])))
        valid_mask.append(True)
    if not X_list:
        return np.zeros((0, preprocessed.shape[0], s1)), np.array([]), np.array(valid_mask)
    return np.array(X_list), np.array(y_list), np.array(valid_mask)


# ── Session Parser ─────────────────────────────────────────────

def parse_edf_sessions(filepath: str, channel_names=None):
    """
    Load EDF → epochs + session metadata.
    Returns dict dengan X, y, sessions, preprocessed, char_info, dll.
    """
    if channel_names is None:
        channel_names = P300_CHANNELS

    labels, signals, sfreqs = read_edf(filepath)
    idx = {l: i for i, l in enumerate(labels)}
    fs  = sfreqs[0]

    missing = [c for c in channel_names if c not in idx]
    if missing:
        raise ValueError(f"Channel tidak ditemukan: {missing}")

    ch_indices   = [idx[c] for c in channel_names]
    preprocessed = preprocess_eeg(signals, ch_indices, fs=fs)

    stim_begin  = signals[idx["StimulusBegin"]]
    stim_type   = signals[idx["StimulusType"]]
    onsets      = np.where(np.diff(stim_begin.astype(int))==1)[0]+1
    stim_labels = np.round(stim_type[onsets]).astype(int)

    X, y, valid = extract_epochs(preprocessed, onsets, stim_labels, fs=fs)

    # Char info
    char_chans = [l for l in labels if l.count("_")==2
                  and not l.startswith("EEG") and not l.startswith("ET")]
    char_info  = [{"name": l.split("_")[0],
                   "row":  int(l.split("_")[1]),
                   "col":  int(l.split("_")[2])} for l in char_chans]
    char_arr   = np.array([signals[idx[l]] for l in char_chans])

    cur_target     = signals[idx["CurrentTarget"]]
    ct             = np.round(cur_target[onsets]).astype(int)
    st             = np.round(stim_type[onsets]).astype(int)
    chars_at_onset = char_arr[:, onsets]

    # ET signals (jika ada)
    has_et = "ETLeftEyeGazeX" in idx
    et_signals = None
    if has_et:
        et_signals = {
            "gx_l":  signals[idx["ETLeftEyeGazeX"]],
            "gy_l":  signals[idx["ETLeftEyeGazeY"]],
            "px_l":  signals[idx["ETLeftEyePosX"]],
            "py_l":  signals[idx["ETLeftEyePosY"]],
            "dl":    signals[idx["ETLeftEyeDist"]],
            "pup_l": signals[idx["ETLeftPupilSize"]],
            "vl":    signals[idx["ETLeftEyeValid"]],
        }

    sessions = []
    for s_id in np.unique(ct[ct > 0]):
        sess_mask = ct == s_id
        tgt_flash = sess_mask & (st == 1)
        tgt_idx   = np.where(tgt_flash)[0]
        if len(tgt_idx) == 0:
            continue
        char_counts = chars_at_onset[:, tgt_idx].sum(axis=1)
        n_tf  = len(tgt_idx)

        # Cari karakter target
        match = [ci for j, ci in enumerate(char_info) if char_counts[j] == n_tf]
        if len(match) == 0:
            continue
        if len(match) > 1:
            # Ambil karakter dengan char_count tertinggi yang unik
            max_count = char_counts.max()
            match = [ci for j, ci in enumerate(char_info) if char_counts[j] == max_count]
            if len(match) != 1:
                continue

        # sess_onset_idx selalu di-assign setelah match valid
        sess_onset_idx = np.where(sess_mask)[0]

        # ET baseline
        s_start  = onsets[sess_onset_idx[0]]
        pup_baseline = 4.5
        if has_et:
            vl_bl = et_signals["vl"][s_start:s_start+int(0.5*fs)] > 0.5
            if vl_bl.sum() > 5:
                pup_baseline = float(et_signals["pup_l"][s_start:s_start+int(0.5*fs)][vl_bl].mean())

        sessions.append({
            "session_id":     int(s_id),
            "target":         match[0],
            "onset_indices":  sess_onset_idx,
            "onsets":         onsets[sess_onset_idx],
            "stim_types":     st[sess_onset_idx],
            "chars_at_flash": chars_at_onset[:, sess_onset_idx],
            "char_info":      char_info,
            "pup_baseline":   pup_baseline,
        })
    if len(sessions) == 0:
        print(f"[DEBUG] Tidak ada session valid. ct unique: {np.unique(ct[ct>0])}, total onsets: {len(onsets)}")
    char_rows = np.array([c["row"] for c in char_info])
    char_cols = np.array([c["col"] for c in char_info])

    return {
        "X": X, "y": y,
        "sessions": sessions,
        "preprocessed": preprocessed,
        "onsets": onsets,
        "stim_labels": stim_labels,
        "char_info": char_info,
        "char_rows": char_rows,
        "char_cols": char_cols,
        "fs": fs,
        "has_et": has_et,
        "et_signals": et_signals,
        "n_target":    int(y.sum()),
        "n_nontarget": int((y==0).sum()),
        "channel_names": channel_names,
    }


# ── Eye Tracking Probability ───────────────────────────────────

def compute_et_probability(et_signals: Dict, onsets: np.ndarray,
                             char_rows: np.ndarray, char_cols: np.ndarray,
                             fs: float = 256.0, pup_baseline: float = 4.5) -> np.ndarray:
    """
    Hitung P(char | ET) menggunakan geometri Gaussian.

    Formula per flash:
      gaze → grid position (row_f, col_f) via kalibrasi linear
      P(c|flash) ∝ exp(-w × [(row_c-row_f)²/2σr² + (col_c-col_f)²/2σc²])
      w = 0.5 + 0.3×pupil_attention + 0.2×gaze_confidence

    Akumulasi: log P(c|session) = Σ log P(c|flash_k)
    """
    log_p = np.zeros(N_CHARS)
    for onset in onsets:
        feat = _extract_et_features(et_signals, onset, fs)
        if feat is None:
            continue
        row_f, col_f, sr, sc = _gaze_to_grid(feat)
        weight = 0.5 + 0.3*feat["pupil_attention"] + 0.2*feat["confidence"]
        d2 = ((char_rows-row_f)**2/(2*sr**2) + (char_cols-col_f)**2/(2*sc**2))
        log_p += -weight * d2

    lp = log_p - log_p.max()
    return np.exp(lp) / np.exp(lp).sum()


def _extract_et_features(et_signals, onset, fs, window_s=0.8):
    s0, s1 = onset, onset + int(window_s*fs)
    T = len(et_signals.get("gx_l", []))
    if s1 > T:
        return None
    dl   = et_signals["dl"][s0:s1]
    vl   = et_signals["vl"][s0:s1]
    mask = (vl > 0.5) & (dl > 0)
    if mask.sum() < 5:
        return None
    gx   = float(np.median(et_signals["gx_l"][s0:s1][mask]))
    gy   = float(np.median(et_signals["gy_l"][s0:s1][mask]))
    ep_x = float(np.median(et_signals["px_l"][s0:s1][mask]))
    ep_y = float(np.median(et_signals["py_l"][s0:s1][mask]))
    dist = float(np.median(dl[mask]))
    pup  = et_signals["pup_l"][s0:s1]
    bl   = pup[:max(1,int(0.05*fs))][mask[:max(1,int(0.05*fs))]].mean() \
           if mask[:max(1,int(0.05*fs))].sum() > 0 else 4.5
    plr_m = mask[int(0.2*fs):int(0.7*fs)]
    resp  = pup[int(0.2*fs):int(0.7*fs)][plr_m].mean() if plr_m.sum() > 0 else bl
    constr = max(0.0, bl - resp)
    gx_all = et_signals["gx_l"][s0:s1][mask]
    gy_all = et_signals["gy_l"][s0:s1][mask]
    std_g  = float(np.sqrt(gx_all.std()**2 + gy_all.std()**2)) if len(gx_all) > 3 else 0.1
    return {
        "gaze_x": gx, "gaze_y": gy,
        "eye_pos_x": ep_x, "eye_pos_y": ep_y,
        "dist_mm": dist,
        "confidence": float(min(1.0, mask.sum()/len(mask))),
        "pupil_attention": float(np.tanh(constr/0.3)),
        "gaze_stability": float(np.exp(-std_g/0.05)),
    }


def _gaze_to_grid(feat):
    gx    = np.clip(feat["gaze_x"], -0.1, 1.1)
    gy    = np.clip(feat["gaze_y"], -0.1, 1.1)
    col_f = CALIB_COL_A * gx + CALIB_COL_B
    row_f = CALIB_ROW_A * gy + CALIB_ROW_B
    col_f -= (feat["eye_pos_x"] - 0.5) * 0.8
    row_f -= (feat["eye_pos_y"] - 0.34) * 1.5
    dr    = feat["dist_mm"] / SCREEN_DIST_MM
    stab  = 0.5 + 0.5*feat["gaze_stability"]
    sr    = SIGMA_ROW * dr / stab
    sc    = SIGMA_COL * dr / stab
    return row_f, col_f, sr, sc
