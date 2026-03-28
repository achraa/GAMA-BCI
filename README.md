# P300 Hybrid BCI — EEGNet + Eye Tracking

## Struktur Project
```
p300_mine/
├── backend/
│   ├── main.py
│   ├── db/database.py
│   ├── model/eegnet.py          ← EEGNet + Focal Loss (PyTorch)
│   ├── routes/                  ← upload, train, models, speller, evaluate
│   └── services/                ← preprocessing, training, inference
└── frontend/
    ├── app.py                   ← Main Streamlit app
    ├── upload_page.py
    ├── train_page.py
    ├── models_page.py
    ├── speller_page.py          ← EEG only vs Hybrid comparison
    ├── evaluate_page.py
    ├── channel_page.py          ← Justifikasi 5 channel P300
    └── about_page.py
```

## Install & Jalankan

### Terminal 1 — Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Terminal 2 — Frontend
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Buka Browser
- Dashboard : http://localhost:8501
- API Docs  : http://localhost:8000/docs

## Workflow
1. **Upload Data** → tab "Windows Path" → masukkan path folder EDF
2. **Train** → masukkan Session ID → Start Training (5-fold CV otomatis)
3. **Models** → lihat hasil training, aktifkan model terbaik
4. **Speller** → upload EDF → decode karakter (EEG only + Hybrid jika ada ET)
5. **Evaluate** → upload EDF → lihat AUC, Balanced Accuracy, F1, Recall

## Perbedaan dari Project Temanmu
| Aspek | Project Ini | Project Temanmu |
|-------|------------|----------------|
| Model | EEGNet (ringan) | hybrid_transformer (berat) |
| Channel | 5 (Cz,Pz,P3,P4,Oz) | Semua 16 |
| Framework | PyTorch | PyTorch |
| ET | Probabilistik terpisah | Dalam model |
| Decode | EEG only + Hybrid berdampingan | Hybrid saja |
