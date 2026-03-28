"""frontend/upload_page.py"""
import pandas as pd, requests
import streamlit as st


def render_upload_page(api, header, metric_grid, api_base):
    header("📂 Upload Data EDF",
           "Upload ZIP berisi file EDF, atau daftarkan folder path langsung")

    st.markdown("""
    <div class="step-box">
      <b>Workflow:</b>
      Step 1 — Upload data → dapat <code>session_id</code><br>
      Step 2 — Ke halaman <b>⚙️ Train</b> → masukkan session_id → Start Training<br>
      Step 3 — Ke halaman <b>🔤 Speller</b> → upload EDF → Decode
    </div>""", unsafe_allow_html=True)

    tab_zip, tab_path, tab_history = st.tabs(
        ["📦 Upload ZIP", "🪟 Windows Path", "📋 Session History"])

    with tab_zip:
        st.markdown("#### Upload ZIP Archive")
        st.markdown("""<div class="upload-zone">
          📦 Compress semua file EDF ke dalam satu ZIP<br>
          <small>Flat ZIP atau nested folders — semua *.edf ditemukan otomatis</small>
        </div>""", unsafe_allow_html=True)
        st.markdown("")
        zip_file = st.file_uploader("Pilih file ZIP", type=["zip"], key="zip_up")
        if zip_file and st.button("🚀 Upload & Process", type="primary", use_container_width=True):
            with st.spinner("Mengupload dan mengekstrak..."):
                files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                try:
                    r = requests.post(f"{api_base}/upload/study-folder",
                                      files=files, timeout=300)
                    r.raise_for_status()
                    result = r.json()
                    st.success(f"✅ Session: **{result['session_id']}**")
                    st.session_state["last_session_id"] = result["session_id"]
                    metric_grid([
                        (result.get("n_edf_files","?"), "EDF Files"),
                        (f"{result.get('sfreq','?')} Hz", "Sample Rate"),
                        (str(result.get("n_channels","?")), "Channels"),
                        ("✓" if result.get("has_et") else "✗", "Eye Tracking"),
                    ])
                    st.info(f"💡 Gunakan `session_id = {result['session_id']}` di halaman Train.")
                except Exception as e:
                    st.error(f"Upload gagal: {e}")

    with tab_path:
        st.markdown("#### Daftarkan Folder Path (Windows)")
        st.caption("Gunakan ini jika data EDF sudah ada di komputer — tidak perlu ZIP.")
        st.markdown("""<div class="disclaimer">
          ⚠️ Folder harus bisa diakses oleh proses backend yang sedang berjalan.
        </div>""", unsafe_allow_html=True)
        folder_path = st.text_input(
            "Folder Path",
            value=r"C:\Users\WIN 11\Documents\lomba\code\SE001",
            help="Path folder yang berisi file *.edf")
        if st.button("🚀 Register Path", type="primary", use_container_width=True):
            with st.spinner("Mendaftarkan folder..."):
                result, err = api("post", "/upload/from-path",
                                  json={"folder_path": folder_path})
            if err:
                st.error(f"Gagal: {err}")
            else:
                st.success(f"✅ Session: **{result['session_id']}**")
                st.session_state["last_session_id"] = result["session_id"]
                metric_grid([
                    (result.get("n_edf_files","?"), "EDF Files"),
                    (f"{result.get('sfreq','?')} Hz", "Sample Rate"),
                    (str(result.get("n_channels","?")), "Channels"),
                    ("✓" if result.get("has_et") else "✗", "Eye Tracking"),
                ])
                st.info(f"💡 Gunakan `session_id = {result['session_id']}` di halaman Train.")

    with tab_history:
        st.markdown("#### Session History")
        if st.button("🔄 Refresh"):
            st.rerun()
        data, err = api("get", "/upload/sessions")
        if err:
            st.error(err)
        else:
            sessions = data.get("sessions", [])
            if not sessions:
                st.info("Belum ada session. Upload data dulu.")
            else:
                st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
                del_id = st.text_input("Session ID untuk dihapus")
                if del_id and st.button("🗑 Hapus Session"):
                    _, err = api("delete", f"/upload/sessions/{del_id}")
                    if err: st.error(err)
                    else:   st.success("Dihapus"); st.rerun()
