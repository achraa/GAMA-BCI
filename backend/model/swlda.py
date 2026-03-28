"""model/swlda.py — Stepwise Linear Discriminant Analysis untuk P300 BCI.

SWLDA adalah classifier klasik yang populer di P300 BCI (Krusienski et al. 2006).
Lebih interpretatif dan lebih cepat dari neural network.

Algoritma:
1. Flatten epoch (C×T) → vektor fitur
2. Stepwise feature selection: tambah/hapus fitur berdasarkan F-statistic
3. LDA classifier pada fitur terpilih
"""
import numpy as np
import pickle
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger("p300_api.swlda")


class SWLDA:
    """
    Stepwise Linear Discriminant Analysis untuk P300 BCI.

    Parameters:
        max_features : jumlah maksimum fitur yang diseleksi (default 60)
        p_enter      : p-value threshold untuk masuk model (default 0.1)
        p_remove     : p-value threshold untuk keluar model (default 0.15)
    """

    def __init__(self, max_features: int = 60,
                 p_enter: float = 0.1, p_remove: float = 0.15):
        self.max_features = max_features
        self.p_enter      = p_enter
        self.p_remove     = p_remove
        self.selected_features_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.feature_names_: Optional[list] = None
        self.n_features_in_: int = 0

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        """(N, C, T) → (N, C*T) dengan downsample temporal 4x"""
        X_down = X[:, :, ::4]  # 204 → 51 sampel
        return X_down.reshape(X_down.shape[0], -1)

    def _f_statistic(self, X: np.ndarray, y: np.ndarray,
                      feature_idx: int) -> float:
        """Hitung F-statistic untuk satu fitur."""
        x0 = X[y == 0, feature_idx]
        x1 = X[y == 1, feature_idx]
        n0, n1 = len(x0), len(x1)
        if n0 < 2 or n1 < 2:
            return 0.0
        mean0, mean1 = x0.mean(), x1.mean()
        var0  = x0.var(ddof=1)
        var1  = x1.var(ddof=1)
        pooled_var = ((n0-1)*var0 + (n1-1)*var1) / (n0+n1-2)
        if pooled_var < 1e-12:
            return 0.0
        f = (n0*n1)/(n0+n1) * (mean1-mean0)**2 / pooled_var
        return float(f)

    def _f_to_pvalue(self, f: float, df1: int = 1, df2: int = 100) -> float:
        """Konversi F ke p-value menggunakan scipy."""
        from scipy.stats import f as f_dist
        if f <= 0:
            return 1.0
        return float(1 - f_dist.cdf(f, df1, df2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SWLDA":
        """
        Training SWLDA.

        X: (N, C, T) atau (N, features)
        y: (N,) binary labels
        """
        if X.ndim == 3:
            X_flat = self._flatten(X)
        else:
            X_flat = X.copy()

        self.n_features_in_ = X_flat.shape[1]
        n_total = len(y)
        df2 = n_total - 2

        # Normalize
        self._mu  = X_flat.mean(axis=0)
        self._std = X_flat.std(axis=0) + 1e-8
        X_norm = (X_flat - self._mu) / self._std

        selected = []
        remaining = list(range(self.n_features_in_))

        logger.info("SWLDA: Starting stepwise selection (max=%d features)",
                    self.max_features)

        for step in range(self.max_features):
            # Forward step: cari fitur terbaik untuk ditambahkan
            best_f, best_idx = -1, -1
            for i in remaining:
                f = self._f_statistic(X_norm, y, i)
                if f > best_f:
                    best_f, best_idx = f, i

            if best_idx == -1:
                break

            p_val = self._f_to_pvalue(best_f, 1, df2)
            if p_val > self.p_enter:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Backward step: hapus fitur yang tidak signifikan
            to_remove = []
            for i in selected[:-1]:
                f = self._f_statistic(X_norm, y, i)
                p = self._f_to_pvalue(f, 1, df2)
                if p > self.p_remove:
                    to_remove.append(i)
            for i in to_remove:
                selected.remove(i)
                remaining.append(i)

        if not selected:
            # Fallback: ambil top-60 fitur berdasarkan F-statistic
            logger.warning("SWLDA: No features selected via stepwise, using top-60 by F-stat")
            f_scores = [self._f_statistic(X_norm, y, i)
                        for i in range(self.n_features_in_)]
            selected = list(np.argsort(f_scores)[::-1][:self.max_features])

        self.selected_features_ = np.array(selected)
        logger.info("SWLDA: Selected %d features", len(selected))

        # Fit LDA pada fitur terpilih
        X_sel = X_norm[:, self.selected_features_]
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self._lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        self._lda.fit(X_sel, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilitas P300 (class 1) untuk setiap epoch.
        Output: (N,) array of probabilities in [0,1]
        """
        if X.ndim == 3:
            X_flat = self._flatten(X)
        else:
            X_flat = X.copy()

        X_norm = (X_flat - self._mu) / self._std
        X_sel  = X_norm[:, self.selected_features_]
        proba  = self._lda.predict_proba(X_sel)
        # Return probabilitas class 1 (P300)
        return proba[:, 1].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def save(self, path: str):
        """Simpan model ke file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "SWLDA":
        """Load model dari file."""
        with open(path, "rb") as f:
            return pickle.load(f)
