"""frontend/results_page.py — End-to-end evaluation semua file EDF."""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests as rq
import time


def render_results_page(api, header, metric_grid):
    header("📊 End-to-End Results",
           "Evaluasi akurasi karakter: EEG Only vs ET Only vs Hybrid — per kondisi gaze")

    # ── Model selection ──────────────────────────────────────────
    models_data, _ = api("get", "/models")
    models    = (models_data or {}).get("models", [])
    model_ids = [m["model_id"] for m in models] if models else ["latest"]
    active_idx = next((i for i,m in enumerate(models) if m.get("is_active")), 0)

    with st.form("results_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            model_id = st.selectbox("Model", model_ids, index=active_idx)
        with c2:
            alpha_eeg = st.slider("Bobot EEG (α)", 0.0, 1.0, 0.2, 0.05)
            st.caption(f"Bobot ET: {round(1-alpha_eeg, 2)}")
        with c3:
            max_flash = st.number_input("Max flash (0=semua)", 0, 500, 24, step=12)

        st.markdown("#### Upload File EDF Test")
        st.caption("Upload banyak file sekaligus — sistem akan otomatis kelompokkan per kondisi")
        edf_files = st.file_uploader(
            "Pilih file EDF", type=["edf"],
            accept_multiple_files=True,
            key="batch_edf"
        )

        submitted = st.form_submit_button("🚀 Evaluasi Semua File",
                                           type="primary", use_container_width=True)

    if not submitted:
        st.markdown("""
        <div style="background:#EBF8FF;border:1px solid #90CDF4;border-radius:8px;
                    padding:1rem;font-size:.9rem;color:#2c3e50">
        <b>Cara pakai:</b><br>
        1. Pilih model yang sudah ditraining<br>
        2. Atur bobot EEG vs ET (default α=0.2 karena ET lebih kuat)<br>
        3. Upload semua file EDF dari folder Test sekaligus<br>
        4. Klik Evaluasi — sistem akan kelompokkan otomatis per kondisi gaze
        </div>
        """, unsafe_allow_html=True)
        return

    if not edf_files:
        st.error("Upload minimal 1 file EDF.")
        return

    # ── Proses semua file ────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### Memproses {len(edf_files)} file...")

    progress = st.progress(0)
    results_all = []
    errors = []

    for i, edf_file in enumerate(edf_files):
        fname = edf_file.name
        progress.progress((i+1)/len(edf_files), text=f"Processing: {fname}")

        try:
            files = {"file": (fname, edf_file.getvalue(), "application/octet-stream")}
            data  = {
                "model_id":    model_id,
                "alpha_eeg":   alpha_eeg,
                "alpha_et":    round(1-alpha_eeg, 2),
                "max_flashes": max_flash,
            }
            r = rq.post("http://localhost:8000/speller/decode-edf",
                        files=files, data=data, timeout=120)
            r.raise_for_status()
            result = r.json()

            n = result.get("n_sessions", 0)
            if n == 0:
                errors.append(f"{fname}: tidak ada session")
                continue

            # Tentukan kondisi dari nama file
            condition = _get_condition(fname)

            results_all.append({
                "file":         fname,
                "condition":    condition,
                "n_sessions":   n,
                "acc_eeg":      result.get("accuracy_eeg", 0),
                "acc_et":       result.get("accuracy_et", None),
                "acc_hybrid":   result.get("accuracy_hybrid", None),
                "has_et":       result.get("has_et", False),
                "decoded_eeg":  result.get("decoded_text_eeg", ""),
                "decoded_et":   result.get("decoded_text_et", ""),
                "decoded_hybrid": result.get("decoded_text_hybrid", ""),
                "sessions":     result.get("results", []),
            })

        except Exception as e:
            errors.append(f"{fname}: {e}")

    progress.empty()

    if errors:
        with st.expander(f"⚠️ {len(errors)} file gagal diproses"):
            for e in errors:
                st.caption(e)

    if not results_all:
        st.error("Tidak ada file yang berhasil diproses.")
        return

    df = pd.DataFrame([{
        "File":       r["file"],
        "Kondisi":    r["condition"],
        "Sessions":   r["n_sessions"],
        "EEG Only":   f"{r['acc_eeg']:.1%}",
        "ET Only":    f"{r['acc_et']:.1%}" if r["acc_et"] is not None else "-",
        "Hybrid":     f"{r['acc_hybrid']:.1%}" if r["acc_hybrid"] is not None else "-",
        "Has ET":     "✓" if r["has_et"] else "✗",
        "acc_eeg_f":  r["acc_eeg"],
        "acc_et_f":   r["acc_et"] if r["acc_et"] is not None else None,
        "acc_hyb_f":  r["acc_hybrid"] if r["acc_hybrid"] is not None else None,
    } for r in results_all])

    # ── Summary keseluruhan ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📈 Summary Keseluruhan")

    total_sess = df["Sessions"].sum()
    avg_eeg = df["acc_eeg_f"].mean()
    avg_et  = df["acc_et_f"].dropna().mean() if df["acc_et_f"].notna().any() else None
    avg_hyb = df["acc_hyb_f"].dropna().mean() if df["acc_hyb_f"].notna().any() else None

    cols = st.columns(4)
    cols[0].metric("Total Sessions", str(int(total_sess)))
    cols[1].metric("Avg EEG Only",   f"{avg_eeg:.1%}")
    cols[2].metric("Avg ET Only",    f"{avg_et:.1%}"  if avg_et  else "-")
    cols[3].metric("Avg Hybrid",     f"{avg_hyb:.1%}" if avg_hyb else "-")

    # ── Per-kondisi summary ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🗂️ Hasil per Kondisi Gaze")

    conditions = df["Kondisi"].unique()
    for cond in sorted(conditions):
        cdf = df[df["Kondisi"] == cond]
        n_files = len(cdf)
        n_sess  = cdf["Sessions"].sum()
        c_eeg   = cdf["acc_eeg_f"].mean()
        c_et    = cdf["acc_et_f"].dropna().mean() if cdf["acc_et_f"].notna().any() else None
        c_hyb   = cdf["acc_hyb_f"].dropna().mean() if cdf["acc_hyb_f"].notna().any() else None

        color = _condition_color(cond)
        st.markdown(f"""
        <div class="card" style="border-left:5px solid {color}">
          <h4 style="margin:0;color:{color}">{cond}</h4>
          <small style="color:#718096">{n_files} file · {int(n_sess)} sessions</small>
        </div>""", unsafe_allow_html=True)

        m_cols = st.columns(3)
        m_cols[0].metric("EEG Only",  f"{c_eeg:.1%}")
        m_cols[1].metric("ET Only",   f"{c_et:.1%}"  if c_et  else "-")
        m_cols[2].metric("Hybrid",    f"{c_hyb:.1%}" if c_hyb else "-")

        with st.expander(f"Detail file {cond}"):
            show_df = cdf[["File","Sessions","EEG Only","ET Only","Hybrid"]].copy()
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    # ── Grafik perbandingan ──────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📊 Grafik Perbandingan")

    # Per-kondisi bar chart
    cond_summary = []
    for cond in sorted(conditions):
        cdf = df[df["Kondisi"] == cond]
        cond_summary.append({
            "Kondisi": cond,
            "EEG Only": cdf["acc_eeg_f"].mean(),
            "ET Only":  cdf["acc_et_f"].dropna().mean() if cdf["acc_et_f"].notna().any() else 0,
            "Hybrid":   cdf["acc_hyb_f"].dropna().mean() if cdf["acc_hyb_f"].notna().any() else 0,
        })
    cs_df = pd.DataFrame(cond_summary)

    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(cs_df))
    width = 0.25
    ax.bar(x - width, cs_df["EEG Only"], width, label="EEG Only",
           color="#3182ce", alpha=0.8, edgecolor="white")
    ax.bar(x,          cs_df["ET Only"],  width, label="ET Only",
           color="#38a169", alpha=0.8, edgecolor="white")
    ax.bar(x + width,  cs_df["Hybrid"],   width, label="Hybrid (EEG×ET)",
           color="#d69e2e", alpha=0.8, edgecolor="white")

    ax.axhline(1/72, color="red", lw=1.5, linestyle="--", label=f"Chance ({1/72*100:.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(cs_df["Kondisi"], fontsize=10)
    ax.set_ylabel("Speller Accuracy")
    ax.set_title("End-to-End Speller Accuracy per Kondisi Gaze")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Tambah value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.0f%%",
                     labels=[f"{v*100:.0f}%" for v in [b.get_height() for b in bars]],
                     fontsize=8, padding=2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Per-file line chart
    st.markdown("#### Akurasi per File")
    fig2, ax2 = plt.subplots(figsize=(max(10, len(df)*0.6), 5))
    x2 = np.arange(len(df))
    ax2.plot(x2, df["acc_eeg_f"], "o-", color="#3182ce", lw=2, label="EEG Only", markersize=6)
    if df["acc_et_f"].notna().any():
        ax2.plot(x2, df["acc_et_f"].fillna(0), "s-", color="#38a169",
                 lw=2, label="ET Only", markersize=6)
    if df["acc_hyb_f"].notna().any():
        ax2.plot(x2, df["acc_hyb_f"].fillna(0), "^-", color="#d69e2e",
                 lw=2, label="Hybrid", markersize=6)
    ax2.axhline(1/72, color="red", lw=1.5, linestyle="--",
                label=f"Chance ({1/72*100:.1f}%)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([r["file"].replace("H_01_SE001_","").replace(".edf","")
                          for r in results_all],
                         rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Speller Accuracy")
    ax2.set_title("Akurasi per File EDF")
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # ── Export tabel ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 📋 Tabel Lengkap")
    show_cols = ["File","Kondisi","Sessions","EEG Only","ET Only","Hybrid","Has ET"]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    # Download CSV
    csv = df[show_cols].to_csv(index=False)
    st.download_button("⬇️ Download CSV", csv, "results.csv", "text/csv")


def _get_condition(filename: str) -> str:
    """Ekstrak kondisi gaze dari nama file."""
    fn = filename.upper()
    if "CBGAZEREAL" in fn: return "CBGazeReal"
    if "CBGAZE10"   in fn: return "CBGaze10"
    if "CBGAZE01"   in fn: return "CBGaze01"
    if "CBGAZENO"   in fn: return "CBGazeNo"
    if "CB_TRAIN"   in fn: return "Train"
    return "Unknown"


def _condition_color(condition: str) -> str:
    colors = {
        "CBGazeReal": "#38a169",
        "CBGaze10":   "#3182ce",
        "CBGaze01":   "#805ad5",
        "CBGazeNo":   "#d69e2e",
        "Train":      "#718096",
    }
    return colors.get(condition, "#718096")
