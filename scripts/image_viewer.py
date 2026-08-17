# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/image_viewer.py

import json
import re
import streamlit as st
from pathlib import Path

def natural_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]

ROOT = "/home/xingqianx/.cache/imaginaire4/data/debug/xingqianx/evaluation_results/unigenbench_for_caption_ablation"
PROMPTS_JSON = "/home/xingqianx/.cache/imaginaire4/data/debug/xingqianx/evaluation/unigenbench/v2_1170L_opus4p7_for_ga.json"
EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
CHOICES = [
    ("v2_1170L_opus",           "cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k"),
    ("v2_1170L_opus",           "cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k"),
    ("v2_1170L_opus",           "cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k"),
    ("v2_1170L_opus",           "cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k"),
    ("v2_1170L_opus",           "cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k"),

    ("v2_1170L_opus4p7_mx_tier1",    "cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k"),
    ("v2_1170L_opus4p7_mx_tier1",    "cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k"),
    ("v2_1170L_opus4p7_mx_tier1",    "cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k"),
    ("v2_1170L_opus4p7_mx_tier1",    "cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k"),
    ("v2_1170L_opus4p7_mx_tier1",    "cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k"),

    ("v2_1170L_opus4p7_gc_tier4",    "cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k"),
    ("v2_1170L_opus4p7_gc_tier4",    "cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4",    "cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4",    "cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4",    "cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k"),

    ("v2_1170L_opus4p7_gc_tier4p5",  "cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k"),
    ("v2_1170L_opus4p7_gc_tier4p5",  "cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4p5",  "cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4p5",  "cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k"),
    ("v2_1170L_opus4p7_gc_tier4p5",  "cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k"),

    ("v2_1170L_opus4p7_gc",     "cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k"),
    ("v2_1170L_opus4p7_gc",     "cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k"),
    ("v2_1170L_opus4p7_gc",     "cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k"),
    ("v2_1170L_opus4p7_gc",     "cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k"),
    ("v2_1170L_opus4p7_gc",     "cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k"),
]  # (benchmark, model) pairs; edit to control what appears in dropdowns

CHOICE_LABELS = [f"{b} / {m}" for b, m in CHOICES]

st.set_page_config(page_title="Image Viewer", layout="wide")
st.caption("Local Image Viewer")

root_path = Path(ROOT)

@st.cache_data
def load_prompts():
    with open(PROMPTS_JSON) as f:
        data = json.load(f)
    return {entry["id"]: entry["prompt"] for entry in data["benchmark"]}

def subdirs(path):
    return sorted([d.name for d in path.iterdir() if d.is_dir()], key=natural_key)

if "idx" not in st.session_state:
    st.session_state.idx = 0

with st.sidebar:
    st.markdown("**Model A**")
    label_a = st.selectbox("Model A", CHOICE_LABELS, index=0)
    benchmark_a, model_a = CHOICES[CHOICE_LABELS.index(label_a)]

    st.markdown("**Model B**")
    label_b = st.selectbox("Model B", CHOICE_LABELS, index=min(1, len(CHOICES) - 1))
    benchmark_b, model_b = CHOICES[CHOICE_LABELS.index(label_b)]

    images_a = sorted([p for p in (root_path / benchmark_a / model_a).iterdir() if p.suffix.lower() in EXTS], key=lambda p: natural_key(p.name))
    names = [p.name for p in images_a]

    idx = min(st.session_state.idx, len(names) - 1)
    selected = st.selectbox("Sample", names, index=idx)
    st.session_state.idx = names.index(selected)

sample = names[st.session_state.idx]
prompts = load_prompts()
image_id = re.sub(r"_\d+\.\w+$", "", sample)

col_a, col_b = st.columns(2)
with col_a:
    st.image(str(root_path / benchmark_a / model_a / sample), width="stretch")
    st.caption(label_a)
with col_b:
    path_b = root_path / benchmark_b / model_b / sample
    if path_b.exists():
        st.image(str(path_b), width="stretch")
    st.caption(label_b)

if image_id in prompts:
    st.markdown(f"**Prompt ({image_id}):** {prompts[image_id]}")

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
