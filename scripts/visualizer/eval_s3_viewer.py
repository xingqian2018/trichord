from __future__ import annotations

from pathlib import Path

import streamlit as st

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/eval_s3_viewer.py --server.port 8502

HERE = Path(__file__).resolve().parent

st.navigation(
    [
        st.Page(HERE / "ugb_s3_viewer.py", title="UGB", icon="🧪", url_path="ugb", default=True),
        st.Page(HERE / "cvtg_s3_viewer.py", title="CVTG", icon="🔤", url_path="cvtg"),
    ],
    position="top",
).run()
