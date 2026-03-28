"""frontend/channel_page.py — Justifikasi pemilihan 5 channel P300."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

P300_CH  = ["EEG_Cz","EEG_Pz","EEG_P3","EEG_P4","EEG_Oz"]
NOISE_CH = ["EEG_F3","EEG_Fz","EEG_F4","EEG_T7","EEG_T8"]
ALL_CH   = ["EEG_F3","EEG_Fz","EEG_F4","EEG_T7","EEG_C3","EEG_Cz",
            "EEG_C4","EEG_T8","EEG_CP3","EEG_CP4","EEG_P3","EEG_Pz",
            "EEG_P4","EEG_PO7","EEG_PO8","EEG_Oz"]

ELECTRODES = {
    "Fp1":(-0.3,0.85),"Fp2":(0.3,0.85),
    "F3":(-0.45,0.5),"Fz":(0,0.55),"F4":(0.45,0.5),
    "T7":(-0.9,0),"C3":(-0.45,0),"Cz":(0,0),"C4":(0.45,0),"T8":(0.9,0),
    "CP3":(-0.45,-0.25),"CP4":(0.45,-0.25),
    "P3":(-0.45,-0.5),"Pz":(0,-0.55),"P4":(0.45,-0.5),
    "PO7":(-0.4,-0.75),"PO8":(0.4,-0.75),"Oz":(0,-0.85),
}


def render_channel_page(header):
    header("⚡ Channel Selection",
           "Justifikasi pemilihan 5 channel P300 dari 16 channel tersedia")

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### ✅ Channel yang Dipilih (P300 Area)")
        st.markdown("""
        Berdasarkan literatur P300 (Polich, 2007; Farwell & Donchin, 1988), 
        komponen P300 paling dominan di area **centro-parietal**.
        """)
        info = {
            "EEG_Cz":  ("Central midline", "Peak P300, amplitude tertinggi"),
            "EEG_Pz":  ("Parietal midline", "P300 dominan, banyak dipakai di BCI"),
            "EEG_P3":  ("Parietal kiri", "P300 kuat, konsisten di subjek"),
            "EEG_P4":  ("Parietal kanan", "P300 kuat, simetris dengan P3"),
            "EEG_Oz":  ("Occipital midline", "Visual P300 — penting untuk visual speller"),
        }
        for ch, (region, desc) in info.items():
            st.markdown(f"""
            <div style="background:#F0FFF4;border:1px solid #9AE6B4;border-radius:8px;
                        padding:10px 14px;margin:6px 0">
              <b style="color:#276749;font-family:monospace">{ch.replace('EEG_','')}</b>
              <span style="color:#718096;font-size:.85rem;margin-left:8px">{region}</span><br>
              <span style="color:#4a5568;font-size:.83rem">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("### ❌ Channel yang Dihilangkan")
        noise_info = {
            "EEG_F3/Fz/F4": "Frontal — rentan artifact eye blink & EMG otot dahi",
            "EEG_T7/T8":    "Temporal — artifact otot rahang, noise tinggi",
            "EEG_C3/C4":    "Central lateral — signal P300 lemah",
            "EEG_CP3/CP4":  "Centro-parietal lateral — redundan dengan P3/P4",
            "EEG_PO7/PO8":  "Parieto-occipital lateral — signal lemah",
        }
        for ch, reason in noise_info.items():
            st.markdown(f"""
            <div style="background:#FFF5F5;border:1px solid #FEB2B2;border-radius:8px;
                        padding:8px 14px;margin:4px 0">
              <b style="color:#9b2c2c;font-family:monospace">{ch.replace('EEG_','')}</b>
              <span style="color:#4a5568;font-size:.82rem;margin-left:8px">{reason}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### Peta Elektroda (sistem 10-20)")
        fig, ax = plt.subplots(figsize=(5,5))
        fig.patch.set_facecolor("white")
        ax.add_patch(plt.Circle((0,0),1.0,fill=False,color="#aaa",lw=2))
        ax.plot([0,0],[1.0,1.15],color="#aaa",lw=2)

        for name,(x,y) in ELECTRODES.items():
            ch = f"EEG_{name}"
            if ch in P300_CH:
                clr, sz, zorder = "#38a169", 280, 5
            elif ch in NOISE_CH:
                clr, sz, zorder = "#e53e3e", 180, 4
            else:
                clr, sz, zorder = "#a0aec0", 120, 3
            ax.scatter(x,y,s=sz,c=clr,zorder=zorder,edgecolors="white",lw=1.5)
            ax.text(x,y,name,ha="center",va="center",fontsize=7,
                    color="white",fontweight="bold",zorder=6)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor="#38a169",label="P300 channel (dipilih)"),
            Patch(facecolor="#e53e3e",label="Noise/artifact (dibuang)"),
            Patch(facecolor="#a0aec0",label="Channel lain"),
        ], loc="lower center", fontsize=8, framealpha=0.9)
        ax.set_xlim(-1.3,1.3); ax.set_ylim(-1.3,1.3)
        ax.set_aspect("equal"); ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown("### Mengapa hanya 5 channel?")
        st.markdown("""
        | Aspek | 16 channel | 5 channel |
        |-------|-----------|-----------|
        | Training time | ~4x lebih lama | Lebih cepat |
        | Overfitting risk | Tinggi | Lebih rendah |
        | P300 relevance | Banyak noise | Fokus ke signal |
        | Akurasi | Tidak selalu lebih baik | Kompetitif |
        
        Studi Blankertz et al. (2011) dan Jin et al. (2015) menunjukkan
        bahwa **pemilihan channel yang tepat** lebih penting dari jumlah channel.
        """)
