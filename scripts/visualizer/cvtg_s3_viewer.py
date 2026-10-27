from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
import streamlit as st
from botocore.config import Config
from scipy.optimize import linear_sum_assignment

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/cvtg_s3_viewer.py --server.port 8503

RESULT_ROOT = "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results"
BASELINE_MODELS = [
    "qwen_image_2p1_nope",
    "qwen_image_2p1_pe",
    "gpt_image_2_quality_high",
    "gpt_image_2_quality_low",
    "gpt_image_2p5_sunburst_quality_high",
    "gpt_image_2p5_sunburst_quality_low",
    "gemini_3_1_flash_image",
    "krea2_turbo",
    "flux3_klein",
    "flux3_klein_base",
    "ideogram4_nf4",
    "fibo2_dev",
]
COSMOS_MODELS = [
    "c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg",
]
EXTRA_RUNS = {
    "cvtg500L": [f"{RESULT_ROOT}/cvtg/cvtg500L_opus/cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k"],
    "cvtg102ch": [f"{RESULT_ROOT}/cvtg/cvtg102ch_opus_ascii/cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k"],
}
DISPLAY_NAMES = {
    "cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k": "cosmos3_super_text2image",
}
BENCHMARKS = {
    "cvtg500L": "s3://datasets/cvtg/cvtg_500l/testing_prompt_500l.json",
    "cvtg102ch": "s3://datasets/cvtg/cvtg_102ch/testing_prompt_102ch.json",
}
IMAGES_PER_PAGE = 25
GCS_CREDENTIAL_PATH = os.path.expanduser("~/Project/trichord/credentials/gcs.secret")
PDX_CREDENTIAL_PATH = os.path.expanduser("~/Project/imaginaire4_alt/credentials/pdx_benchmark.secret")
RESULT_PREFIX = "result_cvtg_"
URL_EXPIRES = 3600
GRID_CSS = """<style>
[class*="st-key-grid_"] p { margin: 0; line-height: 1.3; }
[class*="st-key-grid_"] button { min-height: 0; padding: 0; line-height: 1.3; }
[class*="st-key-grid_"] [data-testid="stHorizontalBlock"] { border-bottom: 1px solid rgba(128, 128, 128, 0.15); padding: 1px 0; }
[class*="st-key-head_"] { padding-bottom: 6px; margin-bottom: 4px; border-bottom: 2px solid rgba(128, 128, 128, 0.5); }
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] { border-bottom: none; }
</style>"""

st.set_page_config(page_title="CVTG S3 Viewer", page_icon="🔤", layout="wide")


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


def run_dirs(benchmark: str) -> list[str]:
    return [f"{RESULT_ROOT}/cvtg_for_baseline/{benchmark}/{model}" for model in BASELINE_MODELS] + [
        f"{RESULT_ROOT}/cvtg_for_cosmos3p5/{benchmark}/{model}" for model in COSMOS_MODELS
    ] + EXTRA_RUNS[benchmark]


def run_label(run: str) -> str:
    name = run.rstrip("/").rsplit("/", 1)[-1]
    return DISPLAY_NAMES.get(name, name)


def fetch_run(run: str, client) -> tuple[dict[str, str], list[str], dict | None]:
    bucket, key = split_uri(f"{run.rstrip('/')}/")
    images, result_files = {}, []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key, Delimiter="/"):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(key):]
            stem, _, extension = name.rpartition(".")
            if extension in ("png", "webp", "jpg", "jpeg") and stem.endswith("_0"):
                images[stem[: -len("_0")]] = f"s3://{bucket}/{obj['Key']}"
            elif name.startswith(RESULT_PREFIX) and extension == "json":
                result_files.append(f"s3://{bucket}/{obj['Key']}")
    result_files.sort()
    result = None
    if result_files:
        result_bucket, result_key = split_uri(result_files[-1])
        result = json.loads(client.get_object(Bucket=result_bucket, Key=result_key)["Body"].read())
    return images, result_files, result


@st.cache_resource(show_spinner="Loading all CVTG runs and scores...")
def load_all_runs(runs: tuple[str, ...]) -> dict[str, tuple[dict[str, str], list[str], dict | None]]:
    client = gcs()
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(runs, pool.map(lambda run: fetch_run(run, client), runs)))


@st.cache_resource(show_spinner="Loading CVTG prompts...")
def load_prompts(benchmark: str) -> dict[str, dict]:
    return json.loads(read_object(BENCHMARKS[benchmark], PDX_CREDENTIAL_PATH))


def image_url(uri: str) -> str:
    bucket, key = split_uri(uri)
    return gcs().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES)


def pned_value(entry: dict) -> float:
    pned = entry["pned"]
    return pned["value_sum"] / pned["count"] if isinstance(pned, dict) else pned


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def ned_score(pred: str, gt: str) -> float:
    denom = max(len(pred), len(gt))
    return 1.0 if denom == 0 else 1.0 - edit_distance(pred, gt) / denom


def match_pairs(preds: list[str], gts: list[str]) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    if not preds or not gts:
        return [], list(range(len(gts))), list(range(len(preds)))
    cost = [[1.0 - ned_score(pred, gt) for gt in gts] for pred in preds]
    rows, cols = linear_sum_assignment(cost)
    pairs = sorted((int(g), int(p), 1.0 - cost[p][g]) for p, g in zip(rows, cols))
    unmatched_gt = [g for g in range(len(gts)) if g not in set(cols)]
    unmatched_pred = [p for p in range(len(preds)) if p not in set(rows)]
    return pairs, unmatched_gt, unmatched_pred


def cell_text(text: str) -> str:
    return "`" + text.replace("\n", " ⏎ ").replace("|", "\\|").replace("`", "'") + "`"


def ocr_table(ocr: dict) -> str:
    preds, gts = ocr.get("pred_text", []), ocr.get("gt_text", [])
    pairs, unmatched_gt, unmatched_pred = match_pairs(preds, gts)
    lines = ["| gt | pred | ned |", "|---|---|---|"]
    lines += [f"| {cell_text(gts[g])} | {cell_text(preds[p])} | {score:.2f} |" for g, p, score in pairs]
    lines += [f"| {cell_text(gts[g])} | - | 0.00 |" for g in unmatched_gt]
    lines += [f"| - | {cell_text(preds[p])} | extra |" for p in unmatched_pred]
    return "\n".join(lines)


def reload_all() -> None:
    load_all_runs.clear()


def subset_of(uid: str) -> str:
    return uid.split("_len")[0] if "_len" in uid else "_".join(uid.split("_")[:4])


def score_table(runs: list[str], results: dict[str, dict | None], images: dict[str, dict]) -> pd.DataFrame:
    table = {}
    for run in runs:
        stats = (results[run] or {}).get("stats", {})
        table[run_label(run)] = {
            "pned": stats.get("pned", float("nan")),
            "gned": stats.get("gned", float("nan")),
            "success": stats.get("success_count", "-"),
            "images": str(len(images[run])),
        }
    return pd.DataFrame.from_dict(table, orient="index")


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
            cell.markdown(value if isinstance(value, str) else "-" if pd.isna(value) else f"{value:.4f}")

with st.sidebar:
    benchmark = st.segmented_control("Benchmark", list(BENCHMARKS), default="cvtg500L", required=True)
    runs_key = f"runs_{benchmark}"
    st.session_state.setdefault(runs_key, [])
    st.header("Runs")
    selected_runs = st.multiselect(
        "Result folders",
        run_dirs(benchmark),
        format_func=run_label,
        key=runs_key,
    )

    prompts = load_prompts(benchmark)
    st.header("Filter")
    subset = st.selectbox("Subset", ["all"] + sorted({subset_of(uid) for uid in prompts}), key=f"subset_{benchmark}")
    failed_in = st.selectbox(
        "pned < 1 in run",
        ["none"] + selected_runs,
        format_func=lambda run: run if run == "none" else run_label(run),
        key=f"failed_in_{benchmark}",
    )
    jump = st.text_input("Jump to prompt id (e.g. regular_len2_0000)", "").strip()

    st.header("Layout")
    show_ocr = st.checkbox("Show OCR details", True)
    show_upsampled = st.checkbox("Show upsampled prompt", False)
    st.button("Reload scores from S3", on_click=reload_all)

loaded = load_all_runs(tuple(run for name in BENCHMARKS for run in run_dirs(name)))
scope = run_dirs(benchmark)
all_images = {run: loaded[run][0] for run in scope}
result_files = {run: loaded[run][1] for run in scope}
all_results = {run: loaded[run][2] for run in scope}

st.title(f"🔤 CVTG S3 Viewer · {benchmark}")
st.caption(f"prompts: {BENCHMARKS[benchmark]}")

score_grid(scope, score_table(scope, all_results, all_images), runs_key)
scored = {run_label(run): run_label(files[-1]) for run, files in result_files.items() if files}
if scored:
    st.caption("result files: " + ", ".join(f"{label} → {name}" for label, name in scored.items()))
if not selected_runs:
    st.info("Click a run name above (or pick in the sidebar) to compare images.")
    st.stop()

images = {run: all_images[run] for run in selected_runs}
results = {run: all_results[run] for run in selected_runs}

prompt_ids = sorted(set(prompts) | set().union(*(set(found) for found in images.values())))
if jump:
    prompt_ids = [pid for pid in prompt_ids if pid == jump]
else:
    if subset != "all":
        prompt_ids = [pid for pid in prompt_ids if subset_of(pid) == subset]
    if failed_in != "none":
        breakdown = (results[failed_in] or {}).get("breakdown", {})
        prompt_ids = [pid for pid in prompt_ids if f"{pid}_0" in breakdown and pned_value(breakdown[f"{pid}_0"]) < 1]

per_page = IMAGES_PER_PAGE
pages = max(1, (len(prompt_ids) + per_page - 1) // per_page)
top = st.columns([3, 1])
top[0].markdown(f"**{len(prompt_ids):,}** prompts · {pages} pages")
page = top[1].selectbox("Jump to page", range(1, pages + 1), format_func=lambda p: f"{p} / {pages}")
batch = prompt_ids[(page - 1) * per_page: page * per_page]
if not batch:
    st.info("No prompts match.")
    st.stop()

for prompt_id in batch:
    entry = prompts.get(prompt_id, {})
    st.divider()
    st.markdown(f"### {prompt_id}")
    st.markdown(f"**Prompt:** {entry.get('prompt', '(not in benchmark)')}")
    st.markdown("**Target text:** " + " · ".join(f"`{text}`" for text in entry.get("text_list", [])))
    if show_upsampled and entry.get("prompt_upsampled"):
        with st.expander("upsampled prompt"):
            st.write(entry["prompt_upsampled"])
    columns = st.columns(len(selected_runs))
    for column, run in zip(columns, selected_runs):
        with column:
            ocr = (results[run] or {}).get("breakdown", {}).get(f"{prompt_id}_0")
            score = f" · pned {pned_value(ocr):.2f} · gned {ocr['gned']:.2f}" if ocr else " · -"
            st.markdown(f"**{run_label(run)}**{score}")
            uri = images[run].get(prompt_id)
            if uri:
                st.image(image_url(uri), width="stretch")
            else:
                st.info("missing")
            if show_ocr:
                with st.expander("OCR", expanded=bool(ocr) and pned_value(ocr) < 1):
                    if not ocr:
                        st.markdown("-")
                    else:
                        pned = ocr["pned"]
                        pned_detail = f" ({pned['value_sum']:.2f} / {pned['count']})" if isinstance(pned, dict) else ""
                        st.markdown(f"**pned** {pned_value(ocr):.4f}{pned_detail} · **gned** {ocr['gned']:.4f}")
                        st.markdown(ocr_table(ocr))
                        comments = ocr.get("pred_comments", [])
                        if comments:
                            st.caption("\n\n".join(f"{cell_text(text)} — {comment}" for text, comment in zip(ocr.get("pred_text", []), comments)))
