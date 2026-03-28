"""frontend/train_page.py"""
import time
import pandas as pd
import streamlit as st

P300_CH = ["EEG_F3", "EEG_Fz", "EEG_F4", "EEG_T7", "EEG_C3",
    "EEG_Cz", "EEG_C4", "EEG_T8", "EEG_CP3", "EEG_CP4",
    "EEG_P3", "EEG_Pz", "EEG_P4", "EEG_PO7", "EEG_PO8",
    "EEG_Oz"]
ALL_CH  = ["EEG_F3","EEG_Fz","EEG_F4","EEG_T7","EEG_C3","EEG_Cz","EEG_C4","EEG_T8",
           "EEG_CP3","EEG_CP4","EEG_P3","EEG_Pz","EEG_P4","EEG_PO7","EEG_PO8","EEG_Oz"]


def render_train_page(api, header, metric_grid):
    header("⚙️ Model Training",
           "Training EEGNet dengan 5 channel P300 + Focal Loss untuk handle imbalance")

    last_sid = st.session_state.get("last_session_id", "")
    if last_sid:
        st.markdown(f'<div class="step-box">💡 Session terakhir: <code>{last_sid}</code></div>',
                    unsafe_allow_html=True)

    with st.form("train_form"):
        st.markdown("#### Data Source")
        c1, c2 = st.columns(2)
        with c1:
            session_id = st.text_input("Session ID", value=last_sid,
                                        placeholder="Dari halaman Upload")
        with c2:
            data_path = st.text_input("Atau langsung Data Path",
                                       placeholder=r"C:\...\SE001\Train")

        st.markdown("#### Channel Selection")
        st.info("Default: 5 channel P300 terbaik (Cz, Pz, P3, P4, Oz) — "
                "ringan dan akurat berdasarkan literatur.")
        use_custom = st.checkbox("Custom channel selection", value=False)
        if use_custom:
            channels = st.multiselect("Pilih channel", ALL_CH, default=P300_CH)
        else:
            channels = P300_CH
            st.markdown(f"Channel: **{', '.join(P300_CH)}**")

        st.markdown("#### Model & Training")
        c3, c4 = st.columns(2)
        with c3:
            n_epochs   = st.slider("Epochs", 10, 100, 50, 5)
            n_folds    = st.slider("CV Folds", 3, 10, 5)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        with c4:
            lr          = st.select_slider("Learning Rate",
                          [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3,
                          format_func=lambda x: f"{x:.0e}")
            focal_alpha = st.slider("Focal Loss α", 0.5, 0.99, 0.75, 0.01)
            focal_gamma = st.slider("Focal Loss γ", 0.5, 5.0, 2.0, 0.25)

        st.markdown("""
        <div class="disclaimer">
          ⚠️ Imbalance P300: 1:11 (target:non-target). Focal Loss α=0.75 γ=2.0 
          sudah dikonfigurasi untuk handle ini.
        </div>""", unsafe_allow_html=True)

        confirm = st.checkbox("Saya siap mulai training", value=False)
        submitted = st.form_submit_button("🚀 Start Training", type="primary",
                                           use_container_width=True)

    if submitted:
        if not confirm:
            st.warning("Centang konfirmasi dulu.")
            return
        if not session_id and not data_path:
            st.error("Isi session_id atau data_path.")
            return

        payload = {
            "session_id":    session_id or None,
            "data_path":     data_path or None,
            "channels":      channels,
            "n_epochs":      n_epochs,
            "n_folds":       n_folds,
            "batch_size":    batch_size,
            "learning_rate": lr,
            "focal_alpha":   focal_alpha,
            "focal_gamma":   focal_gamma,
        }
        result, err = api("post", "/train", json=payload)
        if err:
            st.error(f"❌ {err}")
        else:
            st.success(f"✅ Training queued: `{result['job_id']}`")
            st.session_state["last_job_id"] = result["job_id"]

    st.markdown("---")
    st.markdown("#### 📊 Job Status")
    job_id = st.text_input("Job ID", value=st.session_state.get("last_job_id",""))
    if job_id:
        col1, col2 = st.columns([1, 2])
        with col1:
            refresh = st.button("🔄 Refresh")
        with col2:
            auto = st.checkbox("Auto refresh (3s)", value=False)

        if refresh or auto:
            status, err = api("get", f"/train/{job_id}")
            if not err and status:
                st.session_state[f"ts_{job_id}"] = status

        status = st.session_state.get(f"ts_{job_id}")
        if status:
            s = status.get("status","?")
            colors = {"queued":"badge-gray","running":"badge-orange",
                      "completed":"badge-green","failed":"badge-red"}
            st.markdown(f'<span class="badge {colors.get(s,"badge-gray")}">'
                        f'{s.upper()}</span>', unsafe_allow_html=True)

            if status.get("model_id"):
                st.success(f"Model: **{status['model_id']}**")

            metrics = status.get("metrics",[])
            if metrics:
                st.markdown("**Hasil per-fold:**")
                df = pd.DataFrame(metrics)
                st.dataframe(df, use_container_width=True, hide_index=True)
                if "roc_auc" in df.columns:
                    avg_auc = df["roc_auc"].mean()
                    avg_rec = df["recall"].mean() if "recall" in df.columns else 0
                    metric_grid([
                        (f"{avg_auc:.4f}", "Avg AUC"),
                        (f"{avg_rec:.4f}", "Avg Recall P300"),
                        (f"{df['f1'].mean():.4f}" if 'f1' in df.columns else "-", "Avg F1"),
                    ])

            logs = status.get("logs",[])
            if logs:
                with st.expander("Training logs", expanded=s=="running"):
                    st.text_area("Logs", value="\n".join(logs[-100:]),
                                 height=250, label_visibility="collapsed")

            if status.get("error"):
                st.error(f"Error: {status['error']}")

            if s in {"queued","running"} and auto:
                time.sleep(3); st.rerun()
