"""frontend/speller_page.py — Speller decode: EEGNet, SWLDA, ET, Hybrid."""
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prob_heatmap(p_char, char_info, target=None, pred=None, title="P(char)"):
    ROWS, COLS = 9, 8
    p_grid = np.array(p_char).reshape(ROWS, COLS)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(p_grid, cmap="hot", vmin=0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03)
    for ri in range(ROWS):
        for ci in range(COLS):
            name = char_info[ri*COLS+ci]["name"]
            pv   = p_grid[ri, ci]
            clr  = "white" if pv > p_grid.max()*0.5 else "#555"
            ax.text(ci, ri, f"{name}\n{pv:.3f}",
                    ha="center", va="center", fontsize=6, color=clr)
    if target:
        tr, tc = target["row"]-1, target["col"]-1
        ax.add_patch(plt.Rectangle((tc-0.5,tr-0.5),1,1,lw=3,
                     edgecolor="cyan",facecolor="none",label="Target"))
    if pred:
        pr, pc = pred["row"]-1, pred["col"]-1
        correct = target and pr==target["row"]-1 and pc==target["col"]-1
        ax.add_patch(plt.Rectangle((pc-0.5,pr-0.5),1,1,lw=3,
                     edgecolor="#38a169" if correct else "#e53e3e",
                     facecolor="none",linestyle="--",label="Predicted"))
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
    ax.set_xticklabels(range(1,COLS+1),fontsize=7)
    ax.set_yticklabels(range(1,ROWS+1),fontsize=7)
    if target or pred:
        ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return fig


def render_speller_page(api, header, metric_grid):
    header("🔤 Speller Decode",
           "Perbandingan 5 metode: EEGNet, SWLDA, ET Only, Hybrid EEGNet×ET, Hybrid SWLDA×ET")

    models_data, _ = api("get", "/models")
    models     = (models_data or {}).get("models", [])
    model_ids  = [m["model_id"] for m in models] if models else ["latest"]
    active_idx = next((i for i,m in enumerate(models) if m.get("is_active")), 0)

    # Tampilkan info model aktif jika ada meta
    if models and active_idx < len(models):
        m = models[active_idx]
        meta = m.get("meta") or {}
        if meta.get("eegnet") and meta.get("swlda"):
            e = meta["eegnet"]; s = meta["swlda"]
            better = meta.get("better","?")
            st.markdown(f"""
            <div style="background:#EBF8FF;border:1px solid #90CDF4;border-radius:8px;
                        padding:.8rem 1rem;font-size:.85rem;margin-bottom:1rem">
            <b>Model aktif:</b> {m['model_id']} &nbsp;|&nbsp;
            EEGNet AUC={e.get('auc',0):.3f} &nbsp;|&nbsp;
            SWLDA AUC={s.get('auc',0):.3f} &nbsp;|&nbsp;
            <b>Rekomendasi: {better}</b>
            </div>""", unsafe_allow_html=True)

    with st.form("speller_form"):
        c1, c2 = st.columns([1, 2])
        with c1:
            model_id   = st.selectbox("Model", model_ids, index=active_idx)
            alpha_eeg  = st.slider("Bobot EEG (α)", 0.0, 1.0, 0.2, 0.05)
            alpha_et   = round(1.0 - alpha_eeg, 2)
            st.caption(f"Bobot ET: {alpha_et}")
            max_flash  = st.number_input("Max flash (0=semua)", 0, 500, 24, step=12)
        with c2:
            edf_file = st.file_uploader("Upload file EDF", type=["edf"])
            st.markdown("""<div class="disclaimer">
              ℹ️ File CBGaze* → tampil 5 metode (EEGNet, SWLDA, ET, 2× Hybrid)<br>
              File CBGazeNo/Train → tampil 2 metode (EEGNet, SWLDA)
            </div>""", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Decode", type="primary",
                                           use_container_width=True)

    if not submitted: return
    if not edf_file:
        st.error("Upload file EDF dulu.")
        return

    with st.spinner("Memproses..."):
        import requests as rq
        files = {"file": (edf_file.name, edf_file.getvalue(), "application/octet-stream")}
        data  = {"model_id": model_id, "alpha_eeg": alpha_eeg,
                 "alpha_et": alpha_et, "max_flashes": max_flash}
        try:
            r = rq.post("http://localhost:8000/speller/decode-edf",
                        files=files, data=data, timeout=120)
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            st.error(f"Decode gagal: {e}"); return

    has_et  = result.get("has_et", False)
    n       = result.get("n_sessions", 0)
    results = result.get("results", [])

    if n == 0:
        st.warning("Tidak ada session yang berhasil di-decode.")
        return

    # ── Summary ──────────────────────────────────────────────────
    st.markdown("### 📊 Hasil Keseluruhan")

    acc_eegnet = result.get("accuracy_eegnet", result.get("accuracy_eeg", 0))
    acc_swlda  = result.get("accuracy_swlda", 0)

    if has_et:
        acc_et       = result.get("accuracy_et", 0)
        acc_hyb_eeg  = result.get("accuracy_hybrid_eegnet",
                                   result.get("accuracy_hybrid", 0))
        acc_hyb_swlda = result.get("accuracy_hybrid_swlda", 0)

        metric_grid([
            (f"{acc_eegnet:.1%}", "EEGNet Only"),
            (f"{acc_swlda:.1%}",  "SWLDA Only"),
            (f"{acc_et:.1%}",     "ET Only"),
            (f"{acc_hyb_eeg:.1%}", "Hybrid\nEEGNet×ET"),
            (f"{acc_hyb_swlda:.1%}", "Hybrid\nSWLDA×ET"),
        ])

        # Bar chart perbandingan
        fig, ax = plt.subplots(figsize=(8, 4))
        methods = ["EEGNet", "SWLDA", "ET Only", "Hybrid\nEEGNet×ET", "Hybrid\nSWLDA×ET"]
        accs    = [acc_eegnet, acc_swlda, acc_et, acc_hyb_eeg, acc_hyb_swlda]
        colors  = ["#3182ce","#805ad5","#38a169","#d69e2e","#e53e3e"]
        bars = ax.bar(methods, accs, color=colors, alpha=0.8, edgecolor="white", width=0.6)
        ax.axhline(1/72, color="gray", lw=1.5, linestyle="--",
                   label=f"Chance ({1/72*100:.1f}%)")
        ax.set_ylabel("Speller Accuracy"); ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
        ax.set_title(f"Perbandingan 5 Metode — {n} sessions")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f"{acc:.0%}", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        col1, col2, col3 = st.columns(3)
        col1.metric("Decoded (EEGNet)",        result.get("decoded_text_eegnet",""))
        col2.metric("Decoded (ET)",            result.get("decoded_text_et",""))
        col3.metric("Decoded (Hybrid EEGNet)", result.get("decoded_text_hybrid_eegnet",""))
        col4, col5 = st.columns(2)
        col4.metric("Decoded (SWLDA)",         result.get("decoded_text_swlda",""))
        col5.metric("Decoded (Hybrid SWLDA)",  result.get("decoded_text_hybrid_swlda",""))
    else:
        metric_grid([
            (f"{acc_eegnet:.1%}", "EEGNet Only"),
            (f"{acc_swlda:.1%}",  "SWLDA Only"),
            (f"{1/72*100:.1f}%",  "Chance Level"),
        ])
        col1, col2 = st.columns(2)
        col1.metric("Decoded (EEGNet)", result.get("decoded_text_eegnet",""))
        col2.metric("Decoded (SWLDA)",  result.get("decoded_text_swlda",""))
        st.info("ℹ️ File ini tidak memiliki Eye Tracking.")

    # ── Per-session detail ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Detail Per Session")

    for i, res in enumerate(results):
        target     = res.get("target", {})
        eegnet_pred = res.get("eegnet_pred", {})
        swlda_pred  = res.get("swlda_pred", {})
        eegnet_ok   = res.get("eegnet_correct", False)
        swlda_ok    = res.get("swlda_correct", False)

        label = (f"Session {res['session_id']} | Target: **{target.get('name','?')}** | "
                 f"EEGNet: {'✓' if eegnet_ok else '✗'}{eegnet_pred.get('name','?')} | "
                 f"SWLDA: {'✓' if swlda_ok else '✗'}{swlda_pred.get('name','?')}")
        if has_et:
            hyb_ok = res.get("hybrid_eegnet_correct", False)
            hyb_pred = res.get("hybrid_eegnet_pred", {})
            label += f" | Hybrid: {'✓' if hyb_ok else '✗'}{hyb_pred.get('name','?')}"

        with st.expander(label, expanded=(i==0)):
            char_info = res.get("char_info") or []
            if not char_info: continue

            if has_et:
                # Tampilkan 5 heatmap
                col1, col2, col3 = st.columns(3)
                with col1:
                    fig = _prob_heatmap(res["p_eegnet"], char_info, target,
                                        eegnet_pred, "P(char | EEGNet)")
                    st.pyplot(fig); plt.close()
                with col2:
                    fig = _prob_heatmap(res["p_swlda"], char_info, target,
                                        swlda_pred, "P(char | SWLDA)")
                    st.pyplot(fig); plt.close()
                with col3:
                    fig = _prob_heatmap(res["p_et"], char_info, target,
                                        res.get("et_pred"), "P(char | ET)")
                    st.pyplot(fig); plt.close()
                col4, col5 = st.columns(2)
                with col4:
                    fig = _prob_heatmap(res["p_hybrid_eegnet"], char_info, target,
                                        res.get("hybrid_eegnet_pred"), "Hybrid EEGNet×ET")
                    st.pyplot(fig); plt.close()
                with col5:
                    fig = _prob_heatmap(res["p_hybrid_swlda"], char_info, target,
                                        res.get("hybrid_swlda_pred"), "Hybrid SWLDA×ET")
                    st.pyplot(fig); plt.close()
            else:
                col1, col2 = st.columns(2)
                with col1:
                    fig = _prob_heatmap(res["p_eegnet"], char_info, target,
                                        eegnet_pred, "P(char | EEGNet)")
                    st.pyplot(fig); plt.close()
                with col2:
                    fig = _prob_heatmap(res["p_swlda"], char_info, target,
                                        swlda_pred, "P(char | SWLDA)")
                    st.pyplot(fig); plt.close()

            st.caption(
                f"Valid flash: {res.get('n_valid_flash','?')} | "
                f"EEGNet conf: {res.get('eegnet_confidence',0):.4f} | "
                f"SWLDA conf: {res.get('swlda_confidence',0):.4f}"
                + (f" | Hybrid conf: {res.get('hybrid_eegnet_confidence',0):.4f}"
                   if has_et else ""))
