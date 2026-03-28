"""frontend/evaluate_page.py"""
import streamlit as st


def render_evaluate_page(api, header, metric_grid):
    header("📈 Evaluasi Model", "Evaluasi model pada file EDF")

    models_data, _ = api("get", "/models")
    models = (models_data or {}).get("models", [])
    if not models:
        st.info("Belum ada model. Training dulu.")
        return

    model_ids  = [m["model_id"] for m in models]
    active_idx = next((i for i,m in enumerate(models) if m.get("is_active")), 0)

    with st.form("eval_form"):
        c1, c2 = st.columns(2)
        with c1:
            model_id = st.selectbox("Model", model_ids, index=active_idx)
        with c2:
            edf_file = st.file_uploader("File EDF", type=["edf"])
        submitted = st.form_submit_button("🚀 Evaluasi", type="primary",
                                           use_container_width=True)

    if not submitted: return
    if not edf_file:
        st.error("Upload file EDF dulu.")
        return

    with st.spinner("Mengevaluasi..."):
        import requests as rq
        files = {"file": (edf_file.name, edf_file.getvalue(), "application/octet-stream")}
        try:
            r = rq.post("http://localhost:8000/evaluate/edf",
                        files=files, data={"model_id": model_id}, timeout=120)
            r.raise_for_status()
            result = r.json()
        except Exception as e:
            st.error(f"Evaluasi gagal: {e}")
            return

    st.success("✅ Evaluasi selesai")
    metrics = result.get("metrics", {})
    metric_grid([
        (f"{metrics.get('balanced_accuracy',0):.1%}", "Bal.Acc"),
        (f"{metrics.get('roc_auc',0):.3f}", "AUC-ROC"),
        (f"{metrics.get('f1',0):.1%}", "F1"),
        (metrics.get("imbalance_ratio","-"), "Imbalance Ratio"),
        (str(result.get("n_target","?")), "Target Epochs"),
        (str(result.get("n_nontarget","?")), "Non-target Epochs"),
    ])
