import re
import streamlit as st
from pathlib import Path

def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

ROOT = "/home/xingqianx/.cache/imaginaire4/data/debug/xingqianx/evaluation_results/unigenbench_for_distill/v2_1170L_opus"
EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
SELECTED_FOLDERS = [
    "base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter6k",
    "cosmos3_super_t2i_dmd_dCM_3k_init_iter3k",
    "cosmos3_super_t2i_dmd_dCM_3k_init_iter5k",
    "cosmos3_super_t2i_dmd_dCM_3k_init_iter8k",
    "cosmos3_super_t2i_dmd_dCM_3k_init_iter10k",
    "cosmos3_super_t2i_dmd_default_iter3k",
    "cosmos3_super_t2i_dmd_default_iter5k",
    "cosmos3_super_t2i_dmd_default_iter8k",
    "cosmos3_super_t2i_dmd_default_iter10k",
    "cosmos3_super_t2i_dmd_optim_beta0pt1_iter3k",
    "cosmos3_super_t2i_dmd_optim_beta0pt1_iter5k",
    "cosmos3_super_t2i_dmd_optim_beta0pt1_iter8k",
    "cosmos3_super_t2i_dmd_optim_beta0pt1_iter10k",
    "base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter6k_re",
    "cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_4step_noneg",
    "cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k",
    "base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter8k",
    "base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter4k",
]  # if empty, show all folders

st.set_page_config(page_title="Image Viewer", layout="wide")
st.caption("Local Image Viewer")

root_path = Path(ROOT)

def subdirs(path):
    return sorted([d.name for d in path.iterdir() if d.is_dir()], key=natural_key)

all_models = subdirs(root_path)
models = [m for m in all_models if m in SELECTED_FOLDERS] if SELECTED_FOLDERS else all_models

if "idx" not in st.session_state:
    st.session_state.idx = 0

with st.sidebar:
    model_a = st.selectbox("Model A", models, index=0)
    model_b = st.selectbox("Model B", models, index=min(1, len(models) - 1))

    images_a = sorted([p for p in (root_path / model_a).iterdir() if p.suffix.lower() in EXTS], key=lambda p: natural_key(p.name))
    names = [p.name for p in images_a]

    selected = st.selectbox("Sample", names, index=st.session_state.idx)
    st.session_state.idx = names.index(selected)

sample = names[st.session_state.idx]

col_a, col_b = st.columns(2)
with col_a:
    st.image(str(root_path / model_a / sample), width="stretch")
with col_b:
    path_b = root_path / model_b / sample
    if path_b.exists():
        st.image(str(path_b), width="stretch")

st.components.v1.html("""
<script>
document.addEventListener('keydown', function(e) {
    const key = e.key;
    let label = null;
    if (key === 'ArrowLeft') label = 'Prev';
    else if (key === 'ArrowRight') label = 'Next';
    if (!label) return;
    const buttons = window.parent.document.querySelectorAll('button');
    for (const btn of buttons) {
        if (btn.innerText.includes(label)) { btn.click(); break; }
    }
});
</script>
""", height=0)

col_prev, col_next, _ = st.columns([1, 1, 8])
with col_prev:
    if st.button("◀ Prev"):
        st.session_state.idx = (st.session_state.idx - 1) % len(names)
        st.rerun()
with col_next:
    if st.button("Next ▶"):
        st.session_state.idx = min(len(names) - 1, st.session_state.idx + 1)
        st.rerun()
