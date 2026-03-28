"""services/augmentation.py — EEG Data Augmentation untuk P300.

Augmentasi dilakukan SETELAH read_edf dan preprocessing,
tapi HANYA pada epoch TARGET (P300) untuk handle imbalance 1:11.

Teknik augmentasi:
1. Gaussian Noise       — tambah noise kecil ke sinyal
2. Temporal Shift       — geser sinyal sedikit dalam waktu
3. Channel Dropout      — zero-out channel acak
4. Amplitude Scaling    — scale amplitudo sedikit
5. Time Warping         — stretch/compress waktu lokal

Referensi: Lotte et al. (2018) "A review of classification algorithms
for EEG-based BCIs: a 10 year update"
"""

import numpy as np
from typing import Tuple


def augment_gaussian_noise(epoch: np.ndarray,
                            noise_factor: float = 0.05) -> np.ndarray:
    """
    Tambah Gaussian noise ke epoch.
    noise_factor: std noise relatif terhadap std sinyal
    """
    std = epoch.std()
    noise = np.random.randn(*epoch.shape) * std * noise_factor
    return (epoch + noise).astype(np.float32)


def augment_temporal_shift(epoch: np.ndarray,
                            max_shift: int = 10) -> np.ndarray:
    """
    Geser sinyal dalam domain waktu (circular shift).
    max_shift: maksimum shift dalam sampel (~39ms pada 256Hz)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(epoch, shift, axis=-1).astype(np.float32)


def augment_channel_dropout(epoch: np.ndarray,
                              dropout_rate: float = 0.1) -> np.ndarray:
    """
    Zero-out channel secara acak.
    dropout_rate: probabilitas tiap channel di-zero
    """
    mask = np.random.rand(epoch.shape[0]) > dropout_rate
    result = epoch.copy()
    result[~mask] = 0.0
    return result.astype(np.float32)


def augment_amplitude_scale(epoch: np.ndarray,
                              scale_range: Tuple[float,float] = (0.8, 1.2)) -> np.ndarray:
    """
    Scale amplitudo sinyal secara acak.
    scale_range: range faktor skala
    """
    scale = np.random.uniform(*scale_range)
    return (epoch * scale).astype(np.float32)


def augment_time_warp(epoch: np.ndarray,
                       sigma: float = 0.2, knot: int = 4) -> np.ndarray:
    """
    Time warping: stretch/compress waktu secara lokal.
    Menggunakan cubic spline interpolation.
    """
    from scipy.interpolate import CubicSpline
    n_ch, n_tp = epoch.shape
    orig_steps = np.linspace(0, 1, n_tp)

    # Buat random warp path
    knot_x  = np.linspace(0, 1, knot + 2)
    knot_y  = knot_x + np.random.normal(0, sigma, size=knot + 2)
    knot_y[0] = 0.0; knot_y[-1] = 1.0  # fix endpoints
    cs      = CubicSpline(knot_x, knot_y)
    warped  = np.clip(cs(orig_steps), 0, 1)

    result = np.zeros_like(epoch)
    for c in range(n_ch):
        result[c] = np.interp(orig_steps, warped, epoch[c])
    return result.astype(np.float32)


# ── Main augmentation function ─────────────────────────────────

def augment_epochs(X: np.ndarray, y: np.ndarray,
                   aug_factor: int = 3,
                   techniques: list = None,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augmentasi epoch TARGET (P300) untuk handle imbalance.

    Alur:
    1. Identifikasi epoch target (y==1)
    2. Untuk setiap target epoch, buat aug_factor salinan dengan augmentasi acak
    3. Gabungkan dengan data asli

    Args:
        X          : (N, C, T) — semua epochs
        y          : (N,) — labels
        aug_factor : berapa kali lipat target epochs ditambah
        techniques : list teknik augmentasi yang dipakai
                     default: ["noise", "shift", "scale"]
        random_state: seed untuk reproducibility

    Returns:
        X_aug, y_aug: data setelah augmentasi (shuffled)
    """
    np.random.seed(random_state)

    if techniques is None:
        techniques = ["noise", "shift", "scale"]

    # Ambil epoch target
    target_idx = np.where(y == 1)[0]
    X_target   = X[target_idx]  # (n_target, C, T)
    n_target   = len(X_target)

    if n_target == 0:
        return X, y

    aug_X_list = []
    aug_y_list = []

    # Generate aug_factor salinan per epoch target
    for _ in range(aug_factor):
        for epoch in X_target:
            # Pilih 1-2 teknik secara acak
            n_tech = np.random.randint(1, min(3, len(techniques)) + 1)
            chosen = np.random.choice(techniques, n_tech, replace=False)

            aug_epoch = epoch.copy()
            for tech in chosen:
                if tech == "noise":
                    aug_epoch = augment_gaussian_noise(aug_epoch, 0.05)
                elif tech == "shift":
                    aug_epoch = augment_temporal_shift(aug_epoch, 10)
                elif tech == "scale":
                    aug_epoch = augment_amplitude_scale(aug_epoch, (0.85, 1.15))
                elif tech == "dropout":
                    aug_epoch = augment_channel_dropout(aug_epoch, 0.1)
                elif tech == "warp":
                    aug_epoch = augment_time_warp(aug_epoch, 0.15)

            aug_X_list.append(aug_epoch)
            aug_y_list.append(1)

    if not aug_X_list:
        return X, y

    X_aug_only = np.array(aug_X_list)
    y_aug_only = np.array(aug_y_list)

    # Gabungkan dengan data asli
    X_combined = np.concatenate([X, X_aug_only], axis=0)
    y_combined = np.concatenate([y, y_aug_only], axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_combined))
    return X_combined[perm], y_combined[perm]


def get_class_balance(y: np.ndarray) -> dict:
    """Info distribusi kelas setelah augmentasi."""
    n_target    = int((y == 1).sum())
    n_nontarget = int((y == 0).sum())
    return {
        "n_target":    n_target,
        "n_nontarget": n_nontarget,
        "ratio":       f"1:{n_nontarget // max(1, n_target)}",
        "target_rate": float(n_target / max(1, len(y))),
    }
