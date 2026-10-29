from __future__ import annotations

import json
import os
import io
import random
import tempfile
from pathlib import Path

import boto3
import lance
import pyarrow as pa
import streamlit as st
import yaml
from PIL import Image

GCS_CONFIG_PATH = os.path.expanduser("~/.config/dir/config.yaml")
GCS_CONFIG_PROFILE = "team-gcs-cosmos"
IMAGE_CRED_PATH = os.path.expanduser("~/Project/imaginaire4_sila/credentials/gcs.secret")
FRAGMENT_MAP_PATH = Path(__file__).resolve().parent / "midtrain_fragment_map.json"

LANCE_STORAGE_OPTIONS = {
    "client_max_retries": "15",
    "client_retry_timeout": "300",
    "request_timeout": "3m",
    "connect_timeout": "30s",
}

PBSS_TO_GCS: dict[str, tuple[str, str]] = {
    "logged_images": ("nv-00-10206-images", "logged_images"),
    "annotated_data": ("nv-00-10206-images", "annotated_data"),
    "webdataset": ("nv-00-10206-webdataset-images", ""),
    "debug": ("nv-00-10206-images", "debug"),
}
GCS_NATIVE_BUCKETS = {"nv-00-10206-images", "nv-00-10206-webdataset-images"}

STATUS_COLUMNS = [
    ("filter", "filtering_qwen3vl_fft_8b_v1_aesthetic_score"),
    ("dedup", "dedup_v3_is_duplicate"),
    ("taxonomy", "taxonomy_tag_v0_subcategory"),
    ("embedding", "embedding_qwen3vl_embed_8b_v1_image_embedding_dim"),
    ("caption v2", "captioning_v2_hamid_snah_capbalance_nocot_dense_caption"),
    ("recover caption v2", "captioning_v2_hamid_snah_capbalance_nocot_dense_recover_caption"),
]
GRID_COLUMNS = 4
GRID_MAX_SIDE = 512
DETAIL_MAX_SIDE = 1600

Image.MAX_IMAGE_PIXELS = None
CAPTION_PREVIEW_CHARS = 160

st.set_page_config(page_title="Midtrain Image Viewer", page_icon="🖼️", layout="wide")


@st.cache_resource
def setup_google_auth() -> None:
    with open(GCS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sa_key = json.loads(cfg["gcp"]["service_account_profiles"][GCS_CONFIG_PROFILE]["json_string"])
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(sa_key, tf)
    tf.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name


@st.cache_resource
def make_image_s3_client():
    with open(IMAGE_CRED_PATH) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        endpoint_url=cred["endpoint_url"],
        region_name=cred.get("region_name", ""),
    )


@st.cache_data
def load_fragment_map() -> dict:
    return json.loads(FRAGMENT_MAP_PATH.read_text())


@st.cache_resource(show_spinner="Opening main table...")
def open_dataset(table_uri: str) -> "lance.LanceDataset":
    return lance.dataset(table_uri, storage_options=LANCE_STORAGE_OPTIONS)


def light_columns(schema: pa.Schema) -> list[str]:
    heavy = (pa.types.is_fixed_size_list, pa.types.is_list, pa.types.is_large_list, pa.types.is_binary, pa.types.is_large_binary)
    return [f.name for f in schema if not any(check(f.type) for check in heavy)]


def parse_s3r(uri: str) -> tuple[str, str, int, int]:
    assert uri.startswith("s3r:"), f"Not an s3r URI: {uri}"
    rest = uri[len("s3r:"):]
    rest = rest.split("//", 1)[1]
    rest = rest.rsplit("@", 1)[0]
    bucket_and_path, range_part = rest.rsplit(":", 1)
    bucket, object_path = bucket_and_path.split("/", 1)
    start, length = map(int, range_part.split("-"))
    return bucket, object_path, start, length


@st.cache_data(show_spinner=False, max_entries=512)
def download_image_bytes(s3r_uri: str) -> bytes:
    bucket, obj_path, start, length = parse_s3r(s3r_uri)
    if bucket in GCS_NATIVE_BUCKETS:
        gcs_bucket, gcs_key = bucket, obj_path
    elif bucket in PBSS_TO_GCS:
        gcs_bucket, prefix = PBSS_TO_GCS[bucket]
        gcs_key = f"{prefix}/{obj_path}" if prefix else obj_path
    else:
        raise ValueError(f"Unknown bucket (no GCS mapping): {bucket}")
    resp = make_image_s3_client().get_object(Bucket=gcs_bucket, Key=gcs_key, Range=f"bytes={start}-{start + length - 1}")
    return resp["Body"].read()


@st.cache_data(show_spinner=False, max_entries=1024)
def preview_bytes(s3r_uri: str, max_side: int) -> bytes:
    image = Image.open(io.BytesIO(download_image_bytes(s3r_uri)))
    image.draft("RGB", (max_side, max_side))
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()


def stratified_sample(candidates: list[int], n: int, seed: int) -> list[int]:
    if len(candidates) <= n:
        return candidates
    rng = random.Random(seed)
    bucket_size = len(candidates) / n
    picks = []
    for i in range(n):
        lo = int(i * bucket_size)
        hi = max(lo + 1, min(len(candidates), int((i + 1) * bucket_size)))
        picks.append(candidates[rng.randrange(lo, hi)])
    return picks


@st.cache_data(show_spinner="Loading rows from main table...")
def load_rows(table_uri: str, fragment_id: int, source_dataset: str, row_limit: int, seed: int) -> list[dict]:
    dataset = open_dataset(table_uri)
    fragment = dataset.get_fragment(fragment_id)
    sources = fragment.to_table(columns=["source_dataset"]).column("source_dataset").to_pylist()
    candidates = [i for i, s in enumerate(sources) if s == source_dataset]
    picks = stratified_sample(candidates, row_limit, seed)
    if not picks:
        return []
    table = fragment.take(picks, columns=light_columns(dataset.schema))
    return table.to_pylist()


def caption_items(row: dict) -> list[tuple[str, str]]:
    return [(k, v) for k, v in row.items() if "caption" in k and isinstance(v, str) and v.strip() and not k.endswith(("_model", "_error", "_modified_by", "_category"))]


def status_line(row: dict) -> str:
    return " · ".join(f"{'✅' if row.get(col) is not None else '⬜'} {label}" for label, col in STATUS_COLUMNS)


def show_image(row: dict, max_side: int) -> None:
    uri = row.get("image_s3_range")
    if not uri:
        st.info("No image_s3_range")
        return
    try:
        st.image(preview_bytes(uri, max_side), width="stretch")
    except Exception as e:
        st.error(f"Failed to load image: {e}")


def render_grid(rows: list[dict]) -> None:
    for start in range(0, len(rows), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, row in zip(cols, rows[start : start + GRID_COLUMNS]):
            with col:
                show_image(row, GRID_MAX_SIDE)
                st.caption(f"{row.get('width')}×{row.get('height')} · {status_line(row)}")
                captions = caption_items(row)
                if captions:
                    st.caption(captions[0][1][:CAPTION_PREVIEW_CHARS] + ("…" if len(captions[0][1]) > CAPTION_PREVIEW_CHARS else ""))


def render_detail(rows: list[dict], key: str) -> None:
    if key not in st.session_state:
        st.session_state[key] = 0
    st.session_state[key] = min(st.session_state[key], len(rows) - 1)
    col_prev, col_next, col_jump = st.columns([1, 1, 3])
    with col_prev:
        if st.button("⬅️ Previous", disabled=st.session_state[key] == 0):
            st.session_state[key] -= 1
    with col_next:
        if st.button("Next ➡️", disabled=st.session_state[key] == len(rows) - 1):
            st.session_state[key] += 1
    with col_jump:
        st.session_state[key] = st.slider("Row", 0, len(rows) - 1, st.session_state[key])
    row = rows[st.session_state[key]]
    img_col, meta_col = st.columns([2, 3])
    with img_col:
        show_image(row, DETAIL_MAX_SIDE)
        st.caption(f"original {row.get('width')}×{row.get('height')} · {status_line(row)}")
        for name, text in caption_items(row):
            st.markdown(f"**{name}**")
            st.write(text)
    with meta_col:
        st.markdown("**Metadata (non-null columns)**")
        caption_names = {name for name, _ in caption_items(row)}
        st.json({k: v for k, v in row.items() if v is not None and k not in caption_names}, expanded=True)


def main() -> None:
    st.title("🖼️ Midtrain Image Viewer")
    setup_google_auth()
    fmap = load_fragment_map()
    table_uri = fmap["main_table"]
    datasets = fmap["datasets"]

    with st.sidebar:
        st.header("Dataset")
        names = sorted(datasets, key=lambda n: -datasets[n]["rows"])
        source_dataset = st.selectbox(
            "Midtrain source_dataset",
            names,
            format_func=lambda n: f"{n} ({datasets[n]['rows']:,} rows, {len(datasets[n]['fragments'])} frags)",
        )
        frag_counts = datasets[source_dataset]["fragments"]
        frag_ids = [int(f) for f in frag_counts]
        st.caption(f"{table_uri}\nfragment map built at table version {fmap['table_version']}")

        st.header("Segment")
        frag_key = f"frag_idx::{source_dataset}"
        if frag_key not in st.session_state:
            st.session_state[frag_key] = 0
        if st.button("🎲 Random fragment", disabled=len(frag_ids) == 1):
            st.session_state[frag_key] = random.randrange(len(frag_ids))
        st.session_state[frag_key] = st.number_input(
            f"Fragment index (0-{len(frag_ids) - 1})",
            min_value=0,
            max_value=len(frag_ids) - 1,
            value=min(st.session_state[frag_key], len(frag_ids) - 1),
        )
        fragment_id = frag_ids[st.session_state[frag_key]]
        st.caption(f"fragment_id = {fragment_id} · {frag_counts[str(fragment_id)]:,} rows of {source_dataset}")

        st.header("Sampling")
        row_limit = st.slider("Rows to sample", min_value=4, max_value=96, value=16, step=4)
        if "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = 0
        if st.button("🔀 Resample"):
            st.session_state["sample_seed"] += 1
        view = st.radio("View", ["Grid", "Detail"], horizontal=True)

    rows = load_rows(table_uri, fragment_id, source_dataset, row_limit, st.session_state["sample_seed"])
    if not rows:
        st.warning(f"No {source_dataset} rows in fragment {fragment_id}. The fragment map may be stale; rerun scripts/build_midtrain_fragment_map.py.")
        return
    st.subheader(f"{source_dataset} · fragment {fragment_id} · {len(rows)} sampled row(s)")
    if view == "Grid":
        render_grid(rows)
    else:
        render_detail(rows, f"row_idx::{source_dataset}::{fragment_id}::{st.session_state['sample_seed']}")


if __name__ == "__main__":
    main()
