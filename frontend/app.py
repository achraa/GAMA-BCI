"""frontend/app.py — P300 Hybrid BCI Dashboard."""
import requests, time
import streamlit as st

from upload_page   import render_upload_page
from train_page    import render_train_page
from models_page   import render_models_page
from speller_page  import render_speller_page
from evaluate_page import render_evaluate_page
from results_page  import render_results_page
from channel_page  import render_channel_page
from about_page    import render_about_page

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="P300 Hybrid BCI", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2544 0%, #1a365d 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr { border-color: #2d4a6e; }
.page-header {
    background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 60%, #3182ce 100%);
    padding: 1.4rem 2rem 1.2rem; border-radius: 12px;
    color: white; margin-bottom: 1.4rem;
}
.page-header h1 { color: white; margin: 0; font-size: 1.7rem; font-weight: 700; }
.page-header p  { color: #bee3f8; margin: .3rem 0 0; font-size: .88rem; }
.card {
    background: white; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 1.2rem 1.5rem; box-shadow: 0 1px 6px rgba(0,0,0,.06); margin-bottom: 1rem;
}
.card.green  { border-left: 5px solid #38a169; }
.card.blue   { border-left: 5px solid #3182ce; }
.card.red    { border-left: 5px solid #e53e3e; }
.card.yellow { border-left: 5px solid #d69e2e; }
.badge        { border-radius: 6px; padding: 2px 10px; font-weight: 600;
                font-size: .82rem; display: inline-block; }
.badge-green  { background: #c6f6d5; color: #276749; }
.badge-blue   { background: #bee3f8; color: #2c5282; }
.badge-red    { background: #fed7d7; color: #9b2c2c; }
.badge-orange { background: #feebc8; color: #7b341e; }
.badge-gray   { background: #EDF2F7; color: #4A5568; }
.metric-grid { display: flex; gap: .8rem; flex-wrap: wrap; margin: .8rem 0; }
.metric-box  {
    flex: 1; min-width: 100px; max-width: 160px;
    background: #F7FAFC; border: 1px solid #E2E8F0;
    border-radius: 8px; padding: .7rem 1rem; text-align: center;
}
.metric-box .val { font-size: 1.55rem; font-weight: 700; color: #2d3748; line-height: 1.2; }
.metric-box .lbl { font-size: .7rem; color: #718096; text-transform: uppercase;
                   letter-spacing: .05em; margin-top: 2px; }
.step-box {
    background: #F0FFF4; border: 1px solid #9AE6B4;
    border-radius: 8px; padding: .9rem 1.1rem;
    font-size: .88rem; color: #276749; margin-bottom: .8rem;
}
.disclaimer {
    background: #FFFBEB; border: 1px solid #F6E05E;
    border-radius: 8px; padding: .7rem 1rem;
    font-size: .8rem; color: #744210;
}
.upload-zone {
    background: #EBF8FF; border: 2px dashed #63B3ED;
    border-radius: 10px; padding: 1.5rem; text-align: center; color: #2b6cb0;
}
</style>
""", unsafe_allow_html=True)


def api(method, path, **kwargs):
    try:
        r = getattr(requests, method)(f"{API_BASE}{path}", timeout=120, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Backend tidak berjalan — jalankan: uvicorn main:app --reload"
    except requests.exceptions.HTTPError as e:
        try:    detail = e.response.json().get("detail", str(e))
        except: detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


def header(title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
      <h1>{title}</h1>
      {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def metric_grid(items):
    boxes = "".join(
        f'<div class="metric-box"><div class="val">{v}</div>'
        f'<div class="lbl">{l}</div></div>' for v, l in items)
    st.markdown(f'<div class="metric-grid">{boxes}</div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 P300 Hybrid BCI")
    st.markdown("---")
    selection = st.radio("Navigation", [
        "📂 Upload Data",
        "⚡ Channel Selection",
        "⚙️ Train",
        "🗂️ Models",
        "🔤 Speller",
        "📊 End-to-End Results",
        "📈 Evaluate",
        "📖 About",
    ], label_visibility="collapsed")
    st.markdown("---")

    now = time.time()
    if now - float(st.session_state.get("_hts", 0)) > 5:
        h, _ = api("get", "/health")
        st.session_state["_hv"]  = h
        st.session_state["_hts"] = now
    health = st.session_state.get("_hv")
    st.markdown(
        '<span style="color:#68D391">● API Online</span>' if health
        else '<span style="color:#FC8181">● API Offline</span>',
        unsafe_allow_html=True)

    if st.session_state.get("last_session_id"):
        st.markdown(f"**Session:** `{st.session_state['last_session_id']}`")
    if st.session_state.get("last_job_id"):
        st.markdown(f"**Job:** `{st.session_state['last_job_id']}`")
    st.caption("v1.0 — EEGNet (5-ch) + Eye Tracking")


# ── Page routing ─────────────────────────────────────────────────
kwargs = dict(api=api, header=header, metric_grid=metric_grid)

if   selection == "📂 Upload Data":
    render_upload_page(**kwargs, api_base=API_BASE)
elif selection == "⚡ Channel Selection":
    render_channel_page(header=header)
elif selection == "⚙️ Train":
    render_train_page(**kwargs)
elif selection == "🗂️ Models":
    render_models_page(**kwargs)
elif selection == "🔤 Speller":
    render_speller_page(**kwargs)
elif selection == "📊 End-to-End Results":
    render_results_page(**kwargs)
elif selection == "📈 Evaluate":
    render_evaluate_page(**kwargs)
elif selection == "📖 About":
    render_about_page(header=header)
