from __future__ import annotations

import json
import os

import boto3
import streamlit as st

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/ugb_result_exam.py

BENCHMARK = "v2_1170L_gpt5p5_cosmos3p5_dense_hamid"
BENCHMARK_DIR = os.path.expanduser(
    f"~/.cache/imaginaire4/data/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/{BENCHMARK}"
)
PROMPT_URI = "s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L_gpt5p5_for_cosmos3p5_dense_upsampled_hamid.json"
PROMPT_CREDENTIAL_PATH = os.path.expanduser("~/Project/imaginaire4_alt/credentials/gcs.secret")
IMAGES_PER_PAGE = 100

st.set_page_config(page_title="UGB Result Exam", page_icon="🧪", layout="wide")


@st.cache_resource
def make_s3_client():
    with open(PROMPT_CREDENTIAL_PATH) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        endpoint_url=cred["endpoint_url"],
        region_name=cred.get("region_name", ""),
    )


@st.cache_data(show_spinner="Loading benchmark prompts...")
def load_prompts() -> dict[str, str]:
    bucket, key = PROMPT_URI[len("s3://"):].split("/", 1)
    data = json.loads(make_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read())
    return {str(e["id"]): e.get("upsampled_prompt") or e["prompt"] for e in data["benchmark"]}


def list_subdirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


@st.cache_data(ttl=60)
def scan_run_images(run_dir: str) -> dict[str, str]:
    images = {}
    for name in os.listdir(run_dir):
        if name.endswith("_0.png"):
            images[name[: -len("_0.png")]] = os.path.join(run_dir, name)
    return images


def prompt_id_sort_key(prompt_id: str) -> tuple[str, int]:
    prefix = prompt_id.rstrip("0123456789")
    suffix = prompt_id[len(prefix):]
    return prefix, int(suffix) if suffix else -1


def main() -> None:
    st.title("🧪 UGB Result Exam")

    st.caption(BENCHMARK)

    with st.sidebar:
        st.header("Runs")
        runs = list_subdirs(BENCHMARK_DIR)
        if not runs:
            st.error(f"No run folders under {BENCHMARK_DIR}")
            st.stop()
        selected_runs = st.multiselect("Runs", runs, default=runs)
        if not selected_runs:
            st.stop()

        run_images = {run: scan_run_images(os.path.join(BENCHMARK_DIR, run)) for run in selected_runs}
        for run in selected_runs:
            st.caption(f"{run}: {len(run_images[run])} images")

        prompt_ids = set.union(*(set(images) for images in run_images.values()))
        prompts = load_prompts()

    rows_per_page = max(1, IMAGES_PER_PAGE // len(selected_runs))
    ordered_ids = sorted(prompt_ids, key=prompt_id_sort_key)

    if not ordered_ids:
        st.warning("No prompt ids match the current selection.")
        return

    num_pages = (len(ordered_ids) + rows_per_page - 1) // rows_per_page
    page_key = "page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    st.session_state[page_key] = min(st.session_state[page_key], num_pages - 1)

    col_prev, col_next, col_jump = st.columns([1, 1, 4])
    with col_prev:
        if st.button("⬅️ Previous", disabled=st.session_state[page_key] == 0):
            st.session_state[page_key] -= 1
    with col_next:
        if st.button("Next ➡️", disabled=st.session_state[page_key] == num_pages - 1):
            st.session_state[page_key] += 1
    with col_jump:
        if num_pages > 1:
            st.session_state[page_key] = st.slider("Page", 0, num_pages - 1, st.session_state[page_key])

    page = st.session_state[page_key]
    page_ids = ordered_ids[page * rows_per_page : (page + 1) * rows_per_page]
    st.caption(f"{len(ordered_ids)} prompt ids · page {page + 1}/{num_pages}")

    for prompt_id in page_ids:
        st.divider()
        st.markdown(f"### {prompt_id}")
        prompt_text = prompts.get(prompt_id)
        if prompt_text:
            with st.expander("Prompt", expanded=False):
                st.write(prompt_text)
        columns = st.columns(len(selected_runs))
        for column, run in zip(columns, selected_runs):
            with column:
                st.markdown(f"**{run}**")
                image_path = run_images[run].get(prompt_id)
                if image_path:
                    st.image(image_path, width="stretch")
                else:
                    st.info("missing")


if __name__ == "__main__":
    main()
