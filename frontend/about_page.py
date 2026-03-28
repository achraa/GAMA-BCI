"""frontend/about_page.py"""
import streamlit as st


def render_about_page(header):
    header("📖 About", "P300 Hybrid BCI — EEGNet + Eye Tracking")

    st.graphviz_chart("""
    digraph Pipeline {
        rankdir=LR;
        node [shape=box, style=rounded];
        EDF  [label="EDF File\n(EEG + ET)"];
        Pre  [label="Preprocessing\n(bandpass, notch, CAR)"];
        CH   [label="5 Channel\n(Cz,Pz,P3,P4,Oz)"];
        EEG  [label="EEGNet\n(PyTorch)"];
        ET   [label="ET Pipeline\n(geometri Gaussian)"];
        PEEG [label="P(char|EEG)"];
        PET  [label="P(char|ET)"];
        HYB  [label="Hybrid Fusion\nα×logP_EEG + β×logP_ET"];
        OUT  [label="Karakter Terpilih\n+ Confidence"];
        EDF -> Pre -> CH -> EEG -> PEEG;
        EDF -> ET -> PET;
        PEEG -> HYB;
        PET  -> HYB;
        PEEG -> OUT;
        HYB  -> OUT;
    }
    """)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### 🧠 Model: EEGNet
- Arsitektur CNN kompak (Lawhern et al. 2018)
- Input: (1, 5, 204) — 5 channel × 800ms
- Temporal conv → Depthwise spatial → Separable conv
- **Focal Loss** α=0.75 γ=2.0 untuk handle imbalance 1:11
- **5-Fold Stratified CV** untuk evaluasi
        """)
    with c2:
        st.markdown("""
### 👁️ Eye Tracking Pipeline
- **GazeX/Y** → posisi grid via kalibrasi linear
- **EyePos** → koreksi head-pose bias
- **EyeDist** → scaling uncertainty (σ)
- **PupilSize** → PLR-based attention weight
- 2D Gaussian kernel per flash
- Log product rule accumulation
        """)

    st.markdown("---")
    st.markdown("""
### 📊 2 Mode Decode

| Mode | Formula | Cocok untuk |
|------|---------|-------------|
| **EEG Only** | argmax P(char\\|EEG) | File tanpa eye tracking |
| **Hybrid** | argmax [α·logP_EEG + β·logP_ET] | File CBGaze* (ada ET) |

File `CBGazeNo` → tidak ada gaze (ET uniform) → hasil mirip EEG Only  
File `CBGaze01/10/Real` → ada gaze → Hybrid lebih akurat
    """)

    st.markdown("---")
    st.caption("Dataset: bigP3BCI SE001 | Framework: PyTorch | "
               "Channels: Cz, Pz, P3, P4, Oz | FS: 256Hz | Epoch: 800ms")
