from __future__ import annotations

import ast
import csv
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
import streamlit as st
from botocore.config import Config

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/ugb_s3_viewer.py

BASELINE_DIRS = [
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/qwen_image_2p1_nope",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/qwen_image_2p1_pe",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/gpt_image_2p5_sunburst",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/gpt_image_2p5_sunburst_quality_high",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/krea2_turbo",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/flux3_klein",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/flux3_klein_base",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/ideogram4_nf4",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/fibo2_dev",
]
FEATURED_COSMOS_DIRS = [
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_super_v2p5_iter741260_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_opus/c3_dense_exp009_sft0_iter31000",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_opus/cosmos3_super_text2image_before_uhg",
]
COSMOS_DIRS = FEATURED_COSMOS_DIRS + [
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_super_v2p5_iter741260_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp003_002_pretrain_getty_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp003_003_pretrain_midtrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp004_000_pretrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp004_001_pretrain_mha_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp004_002_pretrain_mha_dense_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp004_003_pretrain_mha_dense_constlr_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_cosmos3p5_dense_hamid/cosmos3p5_ultra_ruofan_20261027_iter30k_best",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_img_a4/cosmos3p5_ultra_ruofan_20261027_iter30k_best",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L_gpt5p5_img_a4/cosmos3_super_text2image_before_uhg",
]
VIEWS = {
    "Baselines": BASELINE_DIRS + FEATURED_COSMOS_DIRS,
    "Cosmos": COSMOS_DIRS,
}
DISPLAY_NAMES = {
    "c3_dense_exp009_sft0_iter31000": "cosmos3_super_text2image",
    "v2_1170L_gpt5p5_img_a4/cosmos3p5_ultra_ruofan_20261027_iter30k_best": "cosmos3p5_ultra_ruofan_20261027_iter30k_best_a4",
    "v2_1170L_gpt5p5_img_a4/cosmos3_super_text2image_before_uhg": "cosmos3_super_text2image_before_uhg_a4",
}
IMAGES_PER_PAGE = 25
UGB_PROMPTS_URI = "s3://datasets/unigenbench/v2/v2_1170L_G3F.csv"
GENERATION_PROMPT_URIS = {
    "v2_1170L_gpt5p5_cosmos3p5_dense_hamid": "s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L_gpt5p5_for_cosmos3p5_dense_upsampled_hamid.json",
    "v2_1170L_opus_cosmos3p5_dense_xingqian": "s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L_opus_for_cosmos3p5_dense_upsampled_xingqian.json",
    "v2_1170L_gpt5p5_img_a4": "s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L_gpt5p5_img_a4.json",
}
GCS_CREDENTIAL_PATH = os.path.expanduser("~/Project/trichord/credentials/gcs.secret")
PDX_CREDENTIAL_PATH = os.path.expanduser("~/Project/imaginaire4_alt/credentials/pdx_benchmark.secret")
RESULT_FILE = "unigenbench_result.json"
HPSV3_PREFIX = "result_hpsv3"
PRIMARY_DIMENSIONS = [
    "Action", "Attribute", "Compound", "Entity Layout", "Grammar", "Logical Reasoning",
    "Relationship", "Style", "Text Generation", "World Knowledge",
]
URL_EXPIRES = 3600
GRID_CSS = """<style>
[class*="st-key-grid_"] p { margin: 0; line-height: 1.3; font-size: 0.8rem; }
[class*="st-key-grid_"] button { min-height: 0; padding: 0; line-height: 1.3; font-size: 0.8rem; }
[class*="st-key-grid_"] [data-testid="stHorizontalBlock"] { border-bottom: 1px solid rgba(128, 128, 128, 0.15); padding: 1px 0; }
[class*="st-key-head_"] { padding-bottom: 6px; margin-bottom: 4px; border-bottom: 2px solid rgba(128, 128, 128, 0.5); }
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] { border-bottom: none; }
[class*="st-key-grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child { border-left: 2px solid rgba(128, 128, 128, 0.5); padding-left: 8px; }
</style>"""

st.set_page_config(page_title="UGB S3 Viewer", page_icon="🧪", layout="wide")


def split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


@st.cache_resource
def s3_client(credential_path: str):
    with open(credential_path) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        endpoint_url=cred["endpoint_url"],
        region_name=cred.get("region_name", "us-east-1"),
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        config=Config(
            s3={"addressing_style": "path"},
            max_pool_connections=32,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def gcs():
    return s3_client(GCS_CREDENTIAL_PATH)


def read_object(uri: str, credential_path: str = GCS_CREDENTIAL_PATH) -> bytes:
    bucket, key = split_uri(uri)
    return s3_client(credential_path).get_object(Bucket=bucket, Key=key)["Body"].read()


def run_uri(run: str) -> str:
    return run.rstrip("/")


def run_benchmark(run: str) -> str:
    return run_uri(run).rsplit("/", 2)[-2]


def run_label(run: str) -> str:
    benchmark, name = run_uri(run).rsplit("/", 2)[-2:]
    return DISPLAY_NAMES.get(f"{benchmark}/{name}", DISPLAY_NAMES.get(name, name))


@st.cache_data(ttl=300, show_spinner="Listing images...")
def list_images(run: str) -> dict[str, str]:
    bucket, key = split_uri(f"{run_uri(run)}/")
    images = {}
    for page in gcs().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key, Delimiter="/"):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(key):]
            stem, _, extension = name.rpartition(".")
            if extension in ("png", "webp", "jpg", "jpeg") and stem.endswith("_0"):
                images[stem[: -len("_0")]] = f"s3://{bucket}/{obj['Key']}"
    return images


def fetch_result(run: str, client) -> dict | None:
    bucket, key = split_uri(f"{run_uri(run)}/{RESULT_FILE}")
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except client.exceptions.NoSuchKey:
        return None


@st.cache_resource(show_spinner="Loading all judge results...")
def load_all_results(runs: tuple[str, ...]) -> dict[str, dict | None]:
    client = gcs()
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(runs, pool.map(lambda run: fetch_result(run, client), runs)))


def fetch_hpsv3(run: str, client) -> dict | None:
    bucket, key = split_uri(f"{run_uri(run)}/{HPSV3_PREFIX}")
    keys = sorted(obj["Key"] for obj in client.list_objects_v2(Bucket=bucket, Prefix=key).get("Contents", []) if obj["Key"].endswith(".json"))
    return json.loads(client.get_object(Bucket=bucket, Key=keys[-1])["Body"].read()) if keys else None


@st.cache_resource(show_spinner="Loading all HPSv3 results...")
def load_all_hpsv3(runs: tuple[str, ...]) -> dict[str, dict | None]:
    client = gcs()
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(runs, pool.map(lambda run: fetch_hpsv3(run, client), runs)))


def reload_all() -> None:
    load_all_results.clear()
    load_all_hpsv3.clear()
    list_images.clear()


@st.cache_data(show_spinner="Loading UGB prompts...")
def load_ugb_prompts() -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(read_object(UGB_PROMPTS_URI, PDX_CREDENTIAL_PATH).decode("utf-8"))))[1:]
    return {str(row[0]): row[1] for row in rows}


@st.cache_data(show_spinner="Loading generation prompts...")
def load_generation_prompts(benchmark: str) -> dict[str, str]:
    uri = GENERATION_PROMPT_URIS.get(benchmark)
    if uri is None:
        return {}
    data = json.loads(read_object(uri))
    entries = data if isinstance(data, list) else data.get("benchmark", [])
    return {str(e["id"]): e.get("upsampled_prompt") or e["prompt"] for e in entries}


@st.cache_data(ttl=300)
def load_run_prompt(run: str, prompt_id: str) -> str | None:
    try:
        return json.loads(read_object(f"{run_uri(run)}/prompts/{prompt_id}.json")).get("render_prompt")
    except gcs().exceptions.NoSuchKey:
        return None


def image_url(uri: str) -> str:
    bucket, key = split_uri(uri)
    return gcs().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES)


def parse_list(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def judgement(result: dict | None, prompt_id: str) -> list[tuple[str, bool, str]]:
    if not result:
        return []
    entry = result.get("breakdown", {}).get(f"{prompt_id}_0.png")
    if not entry:
        return []
    return list(zip(parse_list(entry["testpoints"]), parse_list(entry["verdicts"]), parse_list(entry["analyses"])))


def prompt_id_sort_key(prompt_id: str) -> tuple[str, int]:
    prefix = prompt_id.rstrip("0123456789")
    suffix = prompt_id[len(prefix):]
    return prefix, int(suffix) if suffix else -1


def score_table(runs: list[str], results: dict[str, dict | None], hpsv3: dict[str, dict | None], total: int) -> pd.DataFrame:
    table = {}
    for run in runs:
        result = results[run]
        hps = hpsv3[run]["stats"]["hpsv3"] if hpsv3[run] else float("nan")
        if not result:
            table[run_label(run)] = {key: float("nan") for key in ("all", "orig", "phi")} | {"success": "-"} | {key: float("nan") for key in PRIMARY_DIMENSIONS} | {"hpsv3": hps}
            continue
        stats = result["stats"]
        column = {subset: 100 * stats[subset]["overall_accuracy"] for subset in ("all", "orig", "phi")}
        column["success"] = f"{len(result.get('breakdown', {}))}/{total}"
        for dimension in PRIMARY_DIMENSIONS:
            entry = stats["all"]["big_class_stats"].get(dimension)
            column[dimension] = 100 * entry["accuracy"] if entry else float("nan")
        column["hpsv3"] = hps
        table[run_label(run)] = column
    return pd.DataFrame.from_dict(table, orient="index")


def passes_filter(prompt_id: str, judgements: dict[str, list], dimension: str, failed_in: str, disagree: bool) -> bool:
    if dimension != "any":
        if not any(tp.split(" - ")[0] == dimension for verdicts in judgements.values() for tp, _, _ in verdicts):
            return False
    if failed_in != "none":
        if all(ok for _, ok, _ in judgements.get(failed_in, [])):
            return False
    if disagree:
        scores = {run: sum(ok for _, ok, _ in verdicts) for run, verdicts in judgements.items() if verdicts}
        if len(set(scores.values())) <= 1:
            return False
    return True


def toggle_run(key: str, run: str) -> None:
    runs = st.session_state[key]
    st.session_state[key] = [r for r in runs if r != run] if run in runs else [*runs, run]


def score_grid(runs: list[str], table: pd.DataFrame, key: str) -> None:
    st.html(GRID_CSS)
    grid = st.container(key=f"grid_{key}", gap=None)
    widths = [4] + [1] * len(table.columns)
    header = grid.container(key=f"head_{key}").columns(widths, vertical_alignment="bottom")
    sort_key = f"sort_{key}"
    st.session_state.setdefault(sort_key, table.columns[0])
    for cell, name in zip(header[1:], table.columns):
        arrow = " ▼" if st.session_state[sort_key] == name else ""
        cell.button(f"**{name}**{arrow}", key=f"sortby_{key}_{name}", type="tertiary", on_click=st.session_state.__setitem__, args=(sort_key, name))
    order = pd.to_numeric(table[st.session_state[sort_key]].astype(str).str.split("/").str[0], errors="coerce")
    ranked = order.sort_values(ascending=False, na_position="last", kind="stable").index
    by_label = {run_label(run): run for run in runs}
    for run in [by_label[label] for label in ranked]:
        row = grid.columns(widths, vertical_alignment="center")
        picked = run in st.session_state[key]
        row[0].button(f"{'✅ ' if picked else ''}{run_label(run)}", key=f"pick_{key}_{run}", type="tertiary", on_click=toggle_run, args=(key, run))
        for cell, name in zip(row[1:], table.columns):
            value = table.at[run_label(run), name]
            cell.markdown(value if isinstance(value, str) else "-" if pd.isna(value) else f"{value:.2f}")


with st.sidebar:
    view = st.segmented_control("View", list(VIEWS), default="Baselines", required=True)
    runs_key = f"runs_{view}"
    st.session_state.setdefault(runs_key, [])
    st.header("Runs")
    selected_runs = st.multiselect(
        "Result folders",
        VIEWS[view],
        format_func=lambda run: f"{run_benchmark(run)}/{run_label(run)}",
        key=runs_key,
    )

    st.header("Filter")
    subset = st.radio("Subset", ["all", "orig", "phi"], horizontal=True)
    dimension = st.selectbox("Has testpoint in", ["any"] + PRIMARY_DIMENSIONS)
    failed_in = st.selectbox("Failed a testpoint in run", ["none"] + selected_runs, format_func=lambda run: run if run == "none" else run_label(run), key=f"failed_in_{view}")
    disagree = st.checkbox("Only prompts where runs score differently", False)
    jump = st.text_input("Jump to prompt id (e.g. orig20, phi115)", "").strip()

    st.header("Layout")
    show_verdicts = st.checkbox("Show judge verdicts", True)
    show_generation_prompt = st.checkbox("Show generation prompt per run", False)
    st.button("Reload scores from S3", on_click=reload_all)

all_runs = tuple(dict.fromkeys(run for runs in VIEWS.values() for run in runs))
loaded = load_all_results(all_runs)
loaded_hpsv3 = load_all_hpsv3(all_runs)
all_results = {run: loaded[run] for run in VIEWS[view]}

st.title(f"🧪 UGB S3 Viewer · {view}")
st.caption(f"judge prompts: {UGB_PROMPTS_URI}")

ugb_prompts = load_ugb_prompts()
score_grid(VIEWS[view], score_table(VIEWS[view], all_results, loaded_hpsv3, len(ugb_prompts)), runs_key)
if not selected_runs:
    st.info("Click a run name above (or pick in the sidebar) to compare images.")
    st.stop()

results = {run: all_results[run] for run in selected_runs}
images = {run: list_images(run) for run in selected_runs}

prompt_ids = sorted(set().union(*(set(found) for found in images.values())), key=prompt_id_sort_key)
if jump:
    prompt_ids = [pid for pid in prompt_ids if pid == jump]
elif subset != "all":
    prompt_ids = [pid for pid in prompt_ids if pid.startswith(subset)]

judgements_by_id = {pid: {run: judgement(results[run], pid) for run in selected_runs} for pid in prompt_ids}
if not jump:
    prompt_ids = [pid for pid in prompt_ids if passes_filter(pid, judgements_by_id[pid], dimension, failed_in, disagree)]

per_page = IMAGES_PER_PAGE
pages = max(1, (len(prompt_ids) + per_page - 1) // per_page)
top = st.columns([3, 1])
top[0].markdown(f"**{len(prompt_ids):,}** prompts · {pages} pages")
page = top[1].selectbox("Jump to page", range(1, pages + 1), format_func=lambda p: f"{p} / {pages}")
batch = prompt_ids[(page - 1) * per_page: page * per_page]
if not batch:
    st.info("No prompts match.")
    st.stop()

generation_prompts = {run: load_generation_prompts(run_benchmark(run)) for run in selected_runs}

for prompt_id in batch:
    st.divider()
    st.markdown(f"### {prompt_id}")
    st.markdown(f"**UGB prompt:** {ugb_prompts.get(prompt_id, '(not in v2_1170L)')}")
    columns = st.columns(len(selected_runs))
    for column, run in zip(columns, selected_runs):
        with column:
            verdicts = judgements_by_id[prompt_id][run]
            score = f" · {sum(ok for _, ok, _ in verdicts)}/{len(verdicts)}" if verdicts else " · -"
            st.markdown(f"**{run_label(run)}**{score}")
            uri = images[run].get(prompt_id)
            if uri:
                st.image(image_url(uri), width="stretch")
            else:
                st.info("missing")
            if show_generation_prompt:
                generation_prompt = generation_prompts[run].get(prompt_id) or load_run_prompt(run, prompt_id)
                if generation_prompt:
                    with st.expander("generation prompt"):
                        st.write(generation_prompt)
            if show_verdicts:
                with st.expander("judge verdicts", expanded=any(not ok for _, ok, _ in verdicts)):
                    if not verdicts:
                        st.markdown("-")
                    for testpoint, ok, analysis in verdicts:
                        st.markdown(f"{'✅' if ok else '❌'} **{testpoint}** — {analysis}")
