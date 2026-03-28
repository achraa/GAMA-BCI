"""frontend/models_page.py"""
import streamlit as st


def render_models_page(api, header, metric_grid):
    header("🗂️ Model Registry", "Kelola model EEGNet yang sudah ditraining")
    if st.button("🔄 Refresh"): st.rerun()

    data, err = api("get", "/models")
    if err: st.error(err); return

    models = (data or {}).get("models", [])
    if not models:
        st.info("Belum ada model. Pergi ke **⚙️ Train** untuk membuat model baru.")
        return

    for m in models:
        is_a = m.get("is_active", False)
        badge = '<span class="badge badge-green">● Active</span>' if is_a \
                else '<span class="badge badge-gray">○ Inactive</span>'
        st.markdown(f"""
        <div class="card {'green' if is_a else 'blue'}">
          <h4 style="margin:0">{badge} &nbsp; {m['model_id']}</h4>
          <small style="color:#718096">
            Registered: {str(m.get('registered_at','—'))[:10]} &nbsp;|&nbsp;
            Channels: {', '.join(m.get('channels_used') or [])}
          </small>
        </div>""", unsafe_allow_html=True)

        metric_grid([
            (f"{m.get('avg_bal_acc',0):.1%}", "Bal.Acc"),
            (f"{m.get('avg_roc_auc',0):.3f}", "AUC-ROC"),
            (f"{m.get('avg_f1',0):.1%}", "F1"),
            (f"{m.get('avg_recall',0):.1%}", "Recall P300"),
            (f"{m.get('best_threshold',0.5):.2f}", "Threshold"),
            (str(m.get("n_channels","?")), "Channels"),
        ])

        c1, c2 = st.columns([1, 1])
        with c1:
            if not is_a and st.button("⭐ Activate", key=f"act_{m['model_id']}"):
                _, e = api("post", f"/models/{m['model_id']}/activate")
                if e: st.error(e)
                else: st.rerun()
        with c2:
            if st.button("🗑 Delete", key=f"del_{m['model_id']}"):
                _, e = api("delete", f"/models/{m['model_id']}")
                if e: st.error(e)
                else: st.rerun()
