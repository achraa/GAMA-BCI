"""services/training.py — EEGNet + Focal Loss + CV dengan augmentasi DALAM fold."""
import json, logging, os, uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger("p300_api.training")
MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "./models"))


class TrainingService:
    def __init__(self):
        self._logs: Dict[str, List[str]] = {}

    def _log(self, job_id, msg):
        ts   = datetime.utcnow().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._logs.setdefault(job_id, []).append(line)
        if len(self._logs[job_id]) > 500:
            self._logs[job_id] = self._logs[job_id][-500:]
        logger.info("[%s] %s", job_id, msg)

    def create_job(self, config: Dict, db=None) -> str:
        job_id = str(uuid.uuid4())[:12]
        self._logs[job_id] = []
        self._log(job_id, "Job created.")
        if db:
            from db.database import TrainingJob
            job = TrainingJob(job_id=job_id, status="queued", config=config)
            db.add(job); db.commit()
        return job_id

    def get_logs(self, job_id):
        return self._logs.get(job_id, [])

    def get_status(self, job_id, db=None):
        if not db:
            return None
        from db.database import TrainingJob, JobMetric
        job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
        if not job:
            return None
        metrics = db.query(JobMetric).filter(JobMetric.job_fk == job.id).all()
        return {
            "job_id": job.job_id, "status": job.status,
            "model_id": job.model_id, "error": job.error_msg,
            "created_at":  job.created_at.isoformat()  if job.created_at  else None,
            "started_at":  job.started_at.isoformat()  if job.started_at  else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            "metrics": [{"fold": m.fold, "split": m.split,
                         "balanced_accuracy": m.balanced_accuracy,
                         "roc_auc": m.roc_auc, "f1": m.f1, "recall": m.recall}
                        for m in metrics],
            "logs": self._logs.get(job_id, [])[-100:],
        }

    def run_training(self, job_id, config, db=None):
        import glob
        from services.preprocessing import parse_edf_sessions, P300_CHANNELS

        if db:
            from db.database import TrainingJob
            job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
            if job:
                job.status = "running"; job.started_at = datetime.utcnow(); db.commit()

        try:
            data_path     = config.get("data_path", "")
            channel_names = config.get("channels", P300_CHANNELS)
            n_epochs      = int(config.get("n_epochs", 50))
            n_folds       = int(config.get("n_folds", 5))
            batch_size    = int(config.get("batch_size", 32))
            lr            = float(config.get("learning_rate", 1e-3))
            aug_factor    = int(config.get("aug_factor", 3))
            use_aug       = bool(config.get("use_augmentation", True))

            # ── 1. Load data ASLI (tanpa augmentasi dulu) ──────
            edf_files = sorted(glob.glob(f"{data_path}/**/*.edf", recursive=True))
            if not edf_files:
                edf_files = sorted(glob.glob(f"{data_path}/*.edf"))
            if not edf_files:
                raise ValueError(f"Tidak ada EDF di: {data_path}")

            self._log(job_id, f"Ditemukan {len(edf_files)} file EDF")
            X_all, y_all = [], []
            for fp in edf_files:
                self._log(job_id, f"Loading: {os.path.basename(fp)}")
                try:
                    d = parse_edf_sessions(fp, channel_names)
                    if len(d["X"]) > 0:
                        X_all.append(d["X"]); y_all.append(d["y"])
                except Exception as e:
                    self._log(job_id, f"Skip {os.path.basename(fp)}: {e}")

            if not X_all:
                raise ValueError("Tidak ada epoch yang berhasil di-load")

            # Data asli — BELUM diaugmentasi
            X_orig = np.concatenate(X_all, axis=0)
            y_orig = np.concatenate(y_all, axis=0)
            n_ch, n_tp = X_orig.shape[1], X_orig.shape[2]

            self._log(job_id, f"Data asli: {len(X_orig)} epochs | "
                               f"Target: {int(y_orig.sum())} | "
                               f"Non-target: {int((y_orig==0).sum())} | "
                               f"Ratio 1:{int((y_orig==0).sum()//max(1,y_orig.sum()))}")
            self._log(job_id, f"Augmentasi: {'AKTIF (dalam fold)' if use_aug else 'NONAKTIF'}"
                               + (f" factor={aug_factor}" if use_aug else ""))

            # ── 2. CV dengan augmentasi DALAM fold ─────────────
            self._log(job_id, f"=== EEGNet: {n_folds}-fold CV ===")
            fold_metrics = self._run_cv(
                X_orig, y_orig, job_id, n_folds,
                n_epochs, batch_size, lr,
                use_aug, aug_factor, db)

            avg_auc = float(np.mean([m["roc_auc"] for m in fold_metrics]))
            avg_bal = float(np.mean([m["balanced_accuracy"] for m in fold_metrics]))
            avg_f1  = float(np.mean([m["f1"] for m in fold_metrics]))
            avg_rec = float(np.mean([m["recall"] for m in fold_metrics]))
            self._log(job_id, f"CV Result: AUC={avg_auc:.4f} | "
                               f"Bal.Acc={avg_bal:.4f} | "
                               f"F1={avg_f1:.4f} | Recall={avg_rec:.4f}")

            # ── 3. Train final — augmentasi pada seluruh data ──
            self._log(job_id, "Training model final...")
            from sklearn.model_selection import train_test_split
            X_tr_orig, X_vl, y_tr_orig, y_vl = train_test_split(
                X_orig, y_orig, test_size=0.2, stratify=y_orig, random_state=42)

            # Augmentasi hanya pada training split final
            if use_aug and aug_factor > 0:
                from services.augmentation import augment_epochs
                X_tr, y_tr = augment_epochs(
                    X_tr_orig, y_tr_orig,
                    aug_factor=aug_factor,
                    techniques=["noise", "shift", "scale"],
                    random_state=42)
                self._log(job_id, f"Final train setelah aug: {len(X_tr)} epochs")
            else:
                X_tr, y_tr = X_tr_orig, y_tr_orig

            model, mu, std, _ = self._train_fold(
                X_tr, y_tr, X_vl, y_vl,
                job_id, n_epochs * 2, batch_size, lr)

            # ── 4. Simpan ──────────────────────────────────────
            import torch
            model_id = f"eegnet_{job_id}"
            art_dir  = MODELS_ROOT / model_id
            art_dir.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), art_dir / "model.pt")
            np.save(art_dir / "mu.npy",  mu)
            np.save(art_dir / "std.npy", std)
            with open(art_dir / "config.json", "w") as f:
                json.dump({"channel_names": channel_names,
                            "n_channels": n_ch, "n_times": n_tp,
                            "aug_factor": aug_factor,
                            "aug_in_fold": True}, f, indent=2)

            if db:
                from db.database import RegisteredModel, TrainingJob
                db.query(RegisteredModel).update({"is_active": False})
                rm = RegisteredModel(
                    model_id=model_id, artifact_dir=str(art_dir),
                    is_active=True, job_id=job_id,
                    avg_bal_acc=avg_bal, avg_roc_auc=avg_auc,
                    avg_f1=avg_f1, avg_recall=avg_rec,
                    n_channels=n_ch, n_times=n_tp,
                    channels_used=channel_names)
                db.add(rm)
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = "completed"; job.model_id = model_id
                    job.finished_at = datetime.utcnow(); db.commit()

            self._log(job_id, f"SELESAI | model_id={model_id}")
            self._log(job_id, f"AUC={avg_auc:.4f} | Bal.Acc={avg_bal:.4f} | "
                               f"F1={avg_f1:.4f} | Recall={avg_rec:.4f}")
            return {"model_id": model_id}

        except Exception as e:
            self._log(job_id, f"ERROR: {e}")
            import traceback; self._log(job_id, traceback.format_exc())
            if db:
                from db.database import TrainingJob
                job = db.query(TrainingJob).filter(TrainingJob.job_id == job_id).first()
                if job:
                    job.status = "failed"; job.error_msg = str(e)
                    job.finished_at = datetime.utcnow(); db.commit()
            raise

    def _run_cv(self, X_orig, y_orig, job_id, n_folds,
                n_epochs, batch_size, lr, use_aug, aug_factor, db):
        """
        CV dengan augmentasi DALAM fold:
        - Training fold: data asli + augmentasi
        - Validation fold: data asli SAJA (tidak diaugmentasi)
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score, classification_report

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (tr_idx, vl_idx) in enumerate(skf.split(X_orig, y_orig)):
            self._log(job_id, f"Fold {fold+1}/{n_folds}...")

            X_tr_fold = X_orig[tr_idx]
            y_tr_fold = y_orig[tr_idx]
            X_vl_fold = X_orig[vl_idx]   # ← TIDAK diaugmentasi
            y_vl_fold = y_orig[vl_idx]

            # Augmentasi hanya pada training fold
            if use_aug and aug_factor > 0:
                from services.augmentation import augment_epochs
                X_tr_fold, y_tr_fold = augment_epochs(
                    X_tr_fold, y_tr_fold,
                    aug_factor=aug_factor,
                    techniques=["noise", "shift", "scale"],
                    random_state=fold * 7 + 42)

            model, mu, std, _ = self._train_fold(
                X_tr_fold, y_tr_fold,
                X_vl_fold, y_vl_fold,
                job_id, n_epochs, batch_size, lr)

            import torch
            device = next(model.parameters()).device
            X_vl_n = (X_vl_fold - mu) / (std + 1e-8)
            X_t = torch.FloatTensor(X_vl_n[:, np.newaxis]).to(device)
            with torch.no_grad():
                y_pred = model(X_t).cpu().numpy().flatten()

            y_bin = (y_pred > 0.5).astype(int)
            try:
                auc = float(roc_auc_score(y_vl_fold, y_pred))
            except Exception:
                auc = 0.5
            report = classification_report(
                y_vl_fold, y_bin, output_dict=True, zero_division=0)
            m = {
                "fold": fold + 1, "split": "val",
                "balanced_accuracy": float(
                    report.get("macro avg", {}).get("recall", 0)),
                "roc_auc": auc,
                "f1":      float(report.get("1", {}).get("f1-score", 0)),
                "recall":  float(report.get("1", {}).get("recall", 0)),
            }
            fold_metrics.append(m)
            self._log(job_id, f"  Fold {fold+1}: AUC={auc:.4f} | "
                               f"Recall={m['recall']:.4f} | F1={m['f1']:.4f}")

            if db:
                from db.database import TrainingJob, JobMetric
                job = db.query(TrainingJob).filter(
                    TrainingJob.job_id == job_id).first()
                if job:
                    jm = JobMetric(
                        job_fk=job.id, fold=fold + 1, split="val",
                        balanced_accuracy=m["balanced_accuracy"],
                        roc_auc=m["roc_auc"], f1=m["f1"], recall=m["recall"])
                    db.add(jm); db.commit()

        return fold_metrics

    def _train_fold(self, X_tr, y_tr, X_vl, y_vl,
                     job_id, n_epochs, batch_size, lr):
        import torch, torch.nn as nn, torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.metrics import roc_auc_score
        from model.eegnet import EEGNet, FocalLoss

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu  = X_tr.mean(axis=(0, 2), keepdims=True)
        std = X_tr.std(axis=(0, 2),  keepdims=True) + 1e-8
        X_tr_n = (X_tr - mu) / std
        X_vl_n = (X_vl - mu) / std

        X_tr_t = torch.FloatTensor(X_tr_n[:, np.newaxis]).to(device)
        X_vl_t = torch.FloatTensor(X_vl_n[:, np.newaxis]).to(device)
        y_tr_t  = torch.FloatTensor(y_tr).to(device)

        model     = EEGNet(n_channels=X_tr.shape[1],
                           n_timepoints=X_tr.shape[2]).to(device)
        loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                               batch_size=batch_size, shuffle=True)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        best_auc, best_state, patience = 0.0, None, 0
        for epoch in range(n_epochs):
            model.train()
            for Xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                y_pred = model(X_vl_t).cpu().numpy().flatten()
            try:
                val_auc = float(roc_auc_score(y_vl, y_pred))
            except Exception:
                val_auc = 0.5

            scheduler.step(1 - val_auc)
            if val_auc > best_auc:
                best_auc   = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience   = 0
            else:
                patience  += 1
            if patience >= 10:
                break

        if best_state:
            model.load_state_dict(best_state)
        return model, mu, std, best_auc
