from __future__ import annotations

import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
import streamlit as st
from botocore.config import Config

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/qimagebench_s3_viewer.py --server.port 8503

RUN_ROOTS = [
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench_baseline",
    "s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench",
]
DISPLAY_NAMES = {
    "exp009_sft0_uhq_iter31k": "cosmos3_super_text2image",
    "exp009c_x9c_iter100": "cosmos3_x9c",
    "exp009_sft0_uhq_iter31k_16x9": "cosmos3_super_text2image_16x9",
    "exp009_sft0_uhq_iter31k_mix7": "cosmos3_super_text2image_mixres",
}
PROMPT_ROOT = "s3://nv-00-10206-vfm/debug/xingqianx/evaluation/qimagebench"
PROMPT_SETS = {"cosmos500": f"{PROMPT_ROOT}/cosmos500.json", "arena1000": f"{PROMPT_ROOT}/arena1000.json"}
PARTITIONS = ["cosmos", "arena", "arena_noocr", "arena_ocr"]
SUBSETS = {"cosmos": "cosmos500", "arena": "arena1000", "arena_noocr": "arena1000_noocr", "arena_ocr": "arena1000_ocr"}
LEVEL1 = ["Quality", "Aesthetics", "Alignment", "Real-world Fidelity", "Creative Generation"]
RESULT_FILE = "qimagebench_result.json"
GCS_CREDENTIAL_PATH = os.path.expanduser("~/Project/trichord/credentials/gcs.secret")
IMAGE_PATTERN = re.compile(r"^((cosmos500|arena1000)_\d+)_0\.(png|webp|jpg|jpeg)$")
IMAGES_PER_PAGE = 50
JUDGE_ITEM = re.compile(r"\*\*(.+?)\*\*:\s*(.*?)\s*→\s*Score:\s*(\S+)")
RAW_ICONS = {"0": "❌", "1": "✅", "2": "🌟"}
URL_EXPIRES = 3600
GRID_CSS = """<style>
[class*="st-key-grid_"] p { margin: 0; line-height: 1.3; }
[class*="st-key-grid_"] button { min-height: 0; padding: 0; line-height: 1.3; }
[class*="st-key-grid_"] [data-testid="stHorizontalBlock"] { border-bottom: 1px solid rgba(128, 128, 128, 0.15); padding: 1px 0; }
[class*="st-key-head_"] { padding-bottom: 6px; margin-bottom: 4px; border-bottom: 2px solid rgba(128, 128, 128, 0.5); }
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] { border-bottom: none; }
</style>"""

st.set_page_config(page_title="QImageBench S3 Viewer", page_icon="🖼️", layout="wide")


def split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


@st.cache_resource
def gcs():
    with open(GCS_CREDENTIAL_PATH) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        endpoint_url=cred["endpoint_url"],
        region_name=cred.get("region_name", ""),
        config=Config(s3={"addressing_style": "path"}),
    )


def read_json(uri: str):
    bucket, key = split_uri(uri)
    return json.loads(gcs().get_object(Bucket=bucket, Key=key)["Body"].read())


def run_label(run: str) -> str:
    name = run.rsplit("/", 1)[-1]
    return DISPLAY_NAMES.get(name, name)


def run_group(run: str) -> str:
    root = next(root for root in RUN_ROOTS if run.startswith(f"{root}/"))
    parts = [root.rsplit("/", 1)[-1], *run[len(root) + 1:].split("/")[:-1]]
    return "/".join(parts)


def child_prefixes(bucket: str, key: str) -> tuple[list[str], bool]:
    response = gcs().list_objects_v2(Bucket=bucket, Prefix=key, Delimiter="/", MaxKeys=100)
    has_images = any(IMAGE_PATTERN.match(obj["Key"][len(key):]) for obj in response.get("Contents", []))
    return [p["Prefix"] for p in response.get("CommonPrefixes", [])], has_images


@st.cache_resource(show_spinner="Listing runs...")
def list_runs() -> list[str]:
    runs = []
    for root in RUN_ROOTS:
        bucket, key = split_uri(f"{root}/")
        for child in child_prefixes(bucket, key)[0]:
            grandchildren, has_images = child_prefixes(bucket, child)
            runs += [f"s3://{bucket}/{p.rstrip('/')}" for p in ([child] if has_images else grandchildren)]
    return sorted(runs, key=lambda run: (run_group(run), run_label(run)))


def fetch_run(run: str) -> dict:
    bucket, key = split_uri(f"{run}/")
    images, names = {}, set()
    for page in gcs().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key, Delimiter="/"):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(key):]
            match = IMAGE_PATTERN.match(name)
            if match:
                images[match.group(1)] = f"s3://{bucket}/{obj['Key']}"
            names.add(name)
    config = read_json(f"{run}/config.json") if "config.json" in names else {}
    result = read_json(f"{run}/{RESULT_FILE}") if RESULT_FILE in names else None
    if result:
        result["breakdown"] = {name.rsplit("_", 1)[0]: entry for name, entry in result.get("breakdown", {}).items()}
    return {"images": images, "config": config, "result": result}


@st.cache_resource(show_spinner="Loading all runs...")
def load_all_runs(runs: tuple[str, ...]) -> dict[str, dict]:
    with ThreadPoolExecutor(16) as pool:
        return dict(zip(runs, pool.map(fetch_run, runs)))


@st.cache_resource(show_spinner="Loading prompt sets...")
def load_prompts() -> dict[str, dict]:
    prompts = {}
    for uri in PROMPT_SETS.values():
        for entry in read_json(uri)["benchmark"]:
            prompts[entry["id"]] = entry
    return prompts


@st.cache_resource(show_spinner="Loading generation prompts...")
def load_generation_prompts(uris: tuple[str, ...]) -> dict[str, str]:
    prompts = {}
    for uri in uris:
        for entry in read_json(uri)["benchmark"]:
            prompts[entry["id"]] = entry.get("prompt_upsampled") or entry["prompt"]
    return prompts


def reload_all() -> None:
    list_runs.clear()
    load_all_runs.clear()


def partition_ids(prompts: dict[str, dict]) -> dict[str, list[str]]:
    arena = sorted(pid for pid in prompts if pid.startswith("arena1000"))
    return {
        "cosmos": sorted(pid for pid in prompts if pid.startswith("cosmos500")),
        "arena": arena,
        "arena_noocr": [pid for pid in arena if str(prompts[pid].get("ocr")) == "False"],
        "arena_ocr": [pid for pid in arena if str(prompts[pid].get("ocr")) == "True"],
    }


def parse_list(value) -> list:
    try:
        return ast.literal_eval(value) if isinstance(value, str) else list(value or [])
    except (ValueError, SyntaxError):
        return [value]


def score_table(runs: list[str], loaded: dict[str, dict], ids: list[str], partition: str) -> pd.DataFrame:
    table = {}
    for run in runs:
        result = loaded[run]["result"]
        stats = (result or {}).get("subsets", {}).get(SUBSETS[partition])
        column = {"total": stats["total"] if stats else float("nan")}
        column |= {name: stats["level1"].get(name, float("nan")) if stats else float("nan") for name in LEVEL1}
        column["scored"] = f"{stats['num_images_with_total']}/{len(ids)}" if stats else "-"
        column["images"] = f"{sum(pid in loaded[run]['images'] for pid in ids)}/{len(ids)}"
        column["folder"] = run_group(run)
        table[run_label(run)] = column
    return pd.DataFrame.from_dict(table, orient="index")


def image_entry(loaded: dict[str, dict], run: str, prompt_id: str) -> dict | None:
    result = loaded[run]["result"]
    return result["breakdown"].get(prompt_id) if result else None


def format_value(value) -> str:
    return value if isinstance(value, str) else "-" if pd.isna(value) else f"{value:.2f}"


def level2_table(runs: list[str], loaded: dict[str, dict], partition: str) -> pd.DataFrame:
    table = {}
    for run in runs:
        stats = ((loaded[run]["result"] or {}).get("subsets", {})).get(SUBSETS[partition])
        if stats:
            table[run_label(run)] = {"total": stats["total"]} | {f"{name}/{sub}": value for name in LEVEL1 for sub, value in stats["level2"].get(name, {}).items()}
    return pd.DataFrame.from_dict(table, orient="index").sort_values("total", ascending=False)


def judge_reasons(output: str) -> dict[str, tuple[str, str]]:
    return {match.group(1).strip(): (match.group(2).strip(), match.group(3)) for match in JUDGE_ITEM.finditer(output.split("</think>")[0])}


def number(value, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def show_breakdown(entry: dict, show_na: bool) -> None:
    for name, value in entry.get("level1", {}).items():
        reasons = judge_reasons(entry.get("judge_outputs", {}).get(name, ""))
        lines = [f"**{name} {number(value, 1)}**"]
        for sub, items in entry.get("level3", {}).get(name, {}).items():
            lines.append(f"- *{sub}* {number(entry['level2'][name].get(sub), 0)}")
            for item, score in items.items():
                if score is None and not show_na:
                    continue
                reason, raw = reasons.get(item, ("", "N/A" if score is None else ""))
                icon = RAW_ICONS.get(raw, "➖")
                lines.append(f"    - {icon} **{item}** {number(score, 0)} — {reason}")
        st.markdown("\n".join(lines))


def show_judge_outputs(entry: dict) -> None:
    outputs = entry.get("judge_outputs", {})
    for tab, name in zip(st.tabs(list(outputs)), outputs):
        with tab:
            st.markdown(outputs[name].split("</think>")[0].replace("<think>", "").strip())


def toggle_run(key: str, run: str) -> None:
    runs = st.session_state[key]
    st.session_state[key] = [r for r in runs if r != run] if run in runs else [*runs, run]


def run_grid(runs: list[str], table: pd.DataFrame, key: str) -> None:
    st.html(GRID_CSS)
    grid = st.container(key=f"grid_{key}", gap=None)
    widths = [4] + [1] * (len(table.columns) - 1) + [3]
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
            cell.markdown(format_value(table.at[run_label(run), name]))


def image_url(uri: str) -> str:
    bucket, key = split_uri(uri)
    return gcs().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES)


all_runs = list_runs()
loaded = load_all_runs(tuple(all_runs))
prompts = load_prompts()
partitions = partition_ids(prompts)
runs_key = "runs_qimagebench"
st.session_state.setdefault(runs_key, [])

with st.sidebar:
    partition = st.segmented_control("Partition", PARTITIONS, default="cosmos", required=True)
    st.header("Runs")
    selected_runs = st.multiselect("Result folders", all_runs, format_func=lambda run: f"{run_group(run)}/{run_label(run)}", key=runs_key)

    st.header("Filter")
    ids = partitions[partition]
    top_categories = sorted({parse_list(prompts[pid].get("taxonomy"))[0] for pid in ids if parse_list(prompts[pid].get("taxonomy"))})
    category = st.selectbox("Taxonomy (top level)", ["any"] + top_categories)
    forms = sorted({prompts[pid].get("prompt_form", "") for pid in ids})
    form = st.selectbox("Prompt form", ["any"] + forms)
    jump = st.text_input("Jump to prompt id (e.g. arena1000_0042)", "").strip()
    scored_runs = [run for run in selected_runs if loaded[run]["result"]]
    order_options = ["prompt id", "score spread across selected runs ↓", *[f"score ↑ in {run_label(run)}" for run in scored_runs]]
    order = st.selectbox("Order prompts by", order_options, key=f"order_{partition}")

    st.header("Layout")
    show_meta = st.checkbox("Show prompt metadata", True)
    show_score_details = st.checkbox("Show score breakdown", True)
    show_na = st.checkbox("Include N/A items in breakdown", False)
    show_judge = st.checkbox("Show raw judge outputs", False)
    show_generation_prompt = st.checkbox("Show generation prompt per run", False)
    st.button("Reload runs from S3", on_click=reload_all)

st.title(f"🖼️ QImageBench S3 Viewer · {partition}")
st.caption(f"prompt sets: {PROMPT_ROOT} · image roots: {', '.join(RUN_ROOTS)}")

run_grid(all_runs, score_table(all_runs, loaded, ids, partition), runs_key)
with st.expander(f"Level-2 breakdown · {partition}"):
    st.dataframe(level2_table(all_runs, loaded, partition), column_config={"_index": st.column_config.TextColumn("run", pinned=True)}, height="content")
if not selected_runs:
    st.info("Click a run name above (or pick in the sidebar) to compare images.")
    st.stop()

if jump:
    prompt_ids = [pid for pid in ids if pid == jump]
else:
    prompt_ids = [
        pid for pid in ids
        if (category == "any" or (parse_list(prompts[pid].get("taxonomy")) or [None])[0] == category)
        and (form == "any" or prompts[pid].get("prompt_form") == form)
    ]


def prompt_score(run: str, prompt_id: str) -> float:
    entry = image_entry(loaded, run, prompt_id)
    return entry["score"] if entry and entry.get("score") is not None else float("nan")


if order.startswith("score spread"):
    def spread(prompt_id: str) -> float:
        scores = [score for score in (prompt_score(run, prompt_id) for run in scored_runs) if not pd.isna(score)]
        return max(scores) - min(scores) if len(scores) > 1 else -1
    prompt_ids.sort(key=spread, reverse=True)
elif order.startswith("score ↑"):
    target = next(run for run in scored_runs if order == f"score ↑ in {run_label(run)}")
    prompt_ids.sort(key=lambda pid: (pd.isna(prompt_score(target, pid)), prompt_score(target, pid)))

pages = max(1, (len(prompt_ids) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
top = st.columns([3, 1])
top[0].markdown(f"**{len(prompt_ids):,}** prompts · {pages} pages")
page = top[1].selectbox("Jump to page", range(1, pages + 1), format_func=lambda p: f"{p} / {pages}", key=f"page_{partition}")
batch = prompt_ids[(page - 1) * IMAGES_PER_PAGE: page * IMAGES_PER_PAGE]
if not batch:
    st.info("No prompts match.")
    st.stop()

generation_prompts = {run: load_generation_prompts(tuple(loaded[run]["config"].get("prompts_csv") or PROMPT_SETS.values())) for run in selected_runs} if show_generation_prompt else {}

for prompt_id in batch:
    prompt_entry = prompts[prompt_id]
    st.divider()
    st.markdown(f"### {prompt_id}")
    st.markdown(f"**Prompt:** {prompt_entry['prompt']}")
    if show_meta:
        meta = [
            f"taxonomy: {' › '.join(map(str, parse_list(prompt_entry.get('taxonomy'))))}",
            f"dims: {prompt_entry.get('dims', '-')}",
            f"form: {prompt_entry.get('prompt_form', '-')}",
            f"writer: {prompt_entry.get('writer_model', '-')}",
        ]
        if "ocr" in prompt_entry:
            meta.append(f"ocr: {prompt_entry['ocr']}")
        st.caption(" · ".join(meta))
    columns = st.columns(len(selected_runs))
    for column, run in zip(columns, selected_runs):
        with column:
            entry = image_entry(loaded, run, prompt_id)
            score = f" · {entry['score']:.1f}" if entry and entry.get("score") is not None else f" · {entry['status']}" if entry else " · -"
            st.markdown(f"**{run_label(run)}**{score}")
            uri = loaded[run]["images"].get(prompt_id)
            if uri:
                st.image(image_url(uri), width="stretch")
            else:
                st.info("missing")
            if show_generation_prompt:
                generation_prompt = generation_prompts[run].get(prompt_id)
                if generation_prompt and generation_prompt != prompt_entry["prompt"]:
                    with st.expander("generation prompt"):
                        st.write(generation_prompt)
            if entry and show_score_details:
                with st.expander("score breakdown"):
                    show_breakdown(entry, show_na)
            if entry and show_judge:
                with st.expander("judge outputs"):
                    show_judge_outputs(entry)
