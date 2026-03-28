# GAMA-BCI

A lightweight P300-based Brain-Computer Interface system that combines EEGNet with probabilistic eye tracking for enhanced speller decoding. Built with PyTorch backend (FastAPI) and Streamlit frontend.

## Project Structure

```
GAMA-BCI/
├── backend/
│   ├── main.py                  ← FastAPI application entry point
│   ├── requirements.txt
│   ├── db/
│   │   └── database.py
│   ├── model/
│   │   ├── eegnet.py           ← EEGNet architecture with Focal Loss
│   │   ├── STNN.py
│   │   └── swlda.py
│   ├── routes/                 ← API endpoints
│   │   ├── upload.py           ← Load EDF files
│   │   ├── train.py            ← Training with 5-fold CV
│   │   ├── models.py           ← Model management
│   │   ├── speller.py          ← Character decoding
│   │   └── evaluate.py         ← Performance evaluation
│   ├── services/               ← Business logic
│   │   ├── preprocessing.py    ← Signal filtering & normalization
│   │   ├── training.py         ← Training pipeline
│   │   ├── inference.py        ← Inference engine
│   │   └── augmentation.py     ← Data augmentation
│   ├── models/                 ← Trained model storage
│   ├── data/                   ← Uploaded EDF files
│   └── utils/
│
└── frontend/
    ├── app.py                  ← Main Streamlit dashboard
    ├── requirements.txt
    ├── upload_page.py          ← Data upload interface
    ├── train_page.py           ← Training configuration & monitoring
    ├── models_page.py          ← Model listing & activation
    ├── speller_page.py         ← EEG-only vs Hybrid decoding
    ├── evaluate_page.py        ← Performance metrics (AUC, Accuracy, F1)
    ├── results_page.py         ← Results visualization
    ├── channel_page.py         ← P300 5-channel justification
    └── about_page.py           ← Project information
```

## Installation & Setup

### Prerequisites

- Python 3.9+
- PyTorch (GPU recommended for training)
- BigP3BCI SE001 dataset (EDF format)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment (optional)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn main:app --reload --port 8000
```

The API will be available at:

- Dashboard: http://localhost:8000/docs (interactive API docs)
- Health check: http://localhost:8000/health

### Frontend Setup

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Create and activate virtual environment (optional)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Streamlit dashboard
streamlit run app.py
```

The dashboard will open at: **http://localhost:8501**

## User Workflow

### Step 1: Upload Data

- Go to **"📤 Upload"** tab
- Select **"Windows Path"** option
- Enter the path to your EDF files folder (e.g., `C:\data\bigP3BCI\SE001`)
- Click **"Load Dataset"** to parse and store EDF files

### Step 2: Train Model

- Navigate to **"🎓 Train"** tab
- Enter a **Session ID** (for identifying training run)
- Configure hyperparameters:
  - Epochs, batch size, learning rate
  - Number of CV folds (default: 5-fold)
  - Select model type (EEGNet recommended)
- Click **"Start Training"**
- Monitor real-time training metrics (loss, accuracy, CV scores)
- Once complete, metrics are saved and model is stored with unique ID

### Step 3: Manage Models

- Go to **"🤖 Models"** tab
- View all trained models with:
  - Training date & CV performance metrics
  - Model size & architecture details
- **Activate** the best-performing model for use in speller/evaluation
- Delete models to save storage

### Step 4: Character Decoding (Speller)

- Navigate to **"🔤 Speller"** tab
- Upload a test EDF file
- System decodes characters using:
  - **EEG-only**: Pure EEGNet predictions
  - **Hybrid**: EEGNet + Probabilistic Eye Tracking fusion
- View side-by-side comparisons of both methods
- See decoded sentence and confidence scores

### Step 5: Evaluate Model

- Go to **"📊 Evaluate"** tab
- Upload test EDF file
- Click **"Evaluate"**
- View performance metrics:
  - **AUC** (Area Under Curve)
  - **Balanced Accuracy**
  - **F1 Score**
  - **Recall**
  - Confusion matrix visualization

### Additional Resources

- **📈 Results**: Historical training results & visualizations
- **📍 Channel Info**: P300 electrode placement justification (5 channels: Cz, Pz, P3, P4, Oz)
- **ℹ️ About**: Project information & documentation

## System Architecture

### Backend (FastAPI)

- **Upload Route**: Reads EDF files → extracts channels → stores in database
- **Training Route**: Preprocessing → augmentation → 5-fold CV → model serialization
- **Models Route**: CRUD operations for model management
- **Speller Route**: Real-time inference on new EDF data
- **Evaluate Route**: Batch evaluation with detailed metrics

### Frontend (Streamlit)

- Real-time training monitoring with live plots
- Interactive model management UI
- File upload with progress feedback
- Performance visualization (confusion matrices, ROC curves, metric tables)

### Signal Processing Pipeline

1. **Loading**: EDF → raw signals (channels: Cz, Pz, P3, P4, Oz)
2. **Filtering**: Bandpass (0.5-10 Hz), remove powerline interference
3. **Normalization**: z-score normalization per channel
4. **Augmentation**: Jittering, scaling for training robustness
5. **Windowing**: Extract P300 windows (-0.1s to 0.6s post-stimulus)

### Model Architecture

- **EEGNet**: Compact CNN designed for EEG (depthwise separable convolutions)
- **Loss Function**: Focal Loss for P300/non-P300 imbalance
- **Optimization**: Adam optimizer with learning rate scheduling
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Eye Tracking Integration

- **Probabilistic Fusion**:
  - EEGNet outputs confidence scores (0-1)
  - Eye tracker provides gaze probability
  - Bayesian combination: P(target) = α×P_EEG + (1-α)×P_eye
  - Improves speller accuracy by 5-15% typically

## Key Features

✅ **Lightweight & Efficient**: EEGNet requires minimal compute vs. transformer models
✅ **Full P300 Pipeline**: From raw EDF → trained model → real-time decoding
✅ **Hybrid Fusion**: Combines EEG + Eye Tracking for improved accuracy
✅ **5-fold CV**: Robust cross-validation prevents overfitting
✅ **Web-based UI**: No installation needed beyond Python dependencies
✅ **Model Versioning**: Track & compare multiple trained models
✅ **Dataset**: BigP3BCI SE001 (public benchmark)

## API Endpoints

```
POST   /upload/load          — Load EDF files from path
POST   /upload/status        — Check upload status

POST   /train/start          — Begin training session
GET    /train/progress/{sid} — Get training progress
GET    /train/history        — Training history

GET    /models/list          — List all models
POST   /models/activate/{id} — Set active model
DELETE /models/{id}          — Delete model

POST   /speller/decode       — Decode characters from EDF
POST   /speller/hybrid       — Hybrid decoding (EEG + Eye Tracking)

POST   /evaluate/metrics     — Evaluate on test set
```

See interactive docs at: **http://localhost:8000/docs**

## Configuration

Edit `backend/routes/train.py` or `backend/services/training.py` to customize:

- EEGNet hyperparameters (channels, temporal filters, etc.)
- Training parameters (epochs, batch size, LR schedule)
- Preprocessing settings (filter bands, normalization)
- CV fold count & random seed

## Troubleshooting

**Backend fails to start**

```bash
# Ensure port 8000 is free
netstat -an | grep 8000
# Or use different port
uvicorn main:app --port 8001
```

**Upload fails with "No EDF files found"**

- Verify path exists and contains `.edf` files
- Check file permissions
- Try absolute path instead of relative

**Training runs out of memory**

- Reduce batch size in train_page.py
- Reduce number of CV folds
- Enable garbage collection in training.py

**Speller shows "No active model"**

- Must train & activate a model first on Models page
- Ensure model files are intact in `backend/models/`

## Dataset Information

- **Source**: Mainsah, B., Fleeting, C., Balmat, T., Sellers, E., & Collins, L. (2025). bigP3BCI: An Open, Diverse and Machine Learning Ready P300-based Brain-Computer Interface Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/0byy-ry86
- **Sampling Rate**: 256 Hz
- **Channels**: 16 total (5 used: Cz, Pz, P3, P4, Oz)
- **Sessions**: 1-2 per subject
- **Trials**: ~2400 per session
- **Classes**: 2 (P300 target / non-target)

## Contributors

Developed for International Data Science Challenge 2026 (IDSC2026) - GAMA BCI Team
