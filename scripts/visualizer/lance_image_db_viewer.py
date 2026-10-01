from __future__ import annotations

import json
import os
import random
import tempfile

import boto3
import lance
import streamlit as st
import yaml

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/lance_image_db_viewer.py

GCS_CONFIG_PATH = os.path.expanduser("~/.config/dir/config.yaml")
GCS_CONFIG_PROFILE = "team-gcs-cosmos"

IMAGE_CRED_PATH = os.path.expanduser("~/Project/imaginaire4_sila/credentials/gcs.secret")

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

KNOWN_TABLES: dict[str, str] = {
    "regular / unsplash_lite (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/unsplash_lite_slice_from_maintable_20260915.lance",
    "regular / megalith-10m (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/megalith-10m_slice_from_maintable_20260915.lance",
    "regular / lsdir (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/lsdir_slice_from_maintable_20260915.lance",
    "regular / pexels_residual_trustedK1_v2 (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/pexels_residual_trustedK1_v2_slice_from_maintable_20260915.lance",
    "regular / human_sft (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/human_sft_slice_from_maintable_20260915.lance",
    "regular / aesthetic-train-v2 (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/aesthetic-train-v2_slice_from_maintable_20260915.lance",
    "regular / flickr2k (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/flickr2k_slice_from_maintable_20260915.lance",
    "regular / photo-concept-bucket (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/photo-concept-bucket_slice_from_maintable_20260915.lance",
    "regular / multiaspect-4k-1m (20260915)": "gs://nv-00-10206-vfm/lancedb/image/regular/multiaspect-4k-1m_slice_from_maintable_20260915.lance",
}

MAX_METADATA_COLS_SHOWN = 40

st.set_page_config(page_title="LanceDB Image Viewer", page_icon="🖼️", layout="wide")


@st.cache_resource
def setup_google_auth() -> None:
    with open(GCS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    json_str = cfg["gcp"]["service_account_profiles"][GCS_CONFIG_PROFILE]["json_string"]
    sa_key = json.loads(json_str)
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


def parse_s3r(uri: str) -> tuple[str, str, int, int]:
    assert uri.startswith("s3r:"), f"Not an s3r URI: {uri}"
    rest = uri[len("s3r:"):]
    profile, rest = rest.split("//", 1)
    rest = rest.rsplit("@", 1)[0]
    bucket_and_path, range_part = rest.rsplit(":", 1)
    bucket, object_path = bucket_and_path.split("/", 1)
    start, length = map(int, range_part.split("-"))
    return bucket, object_path, start, length


def download_image_bytes(s3_client, s3r_uri: str) -> bytes:
    bucket, obj_path, start, length = parse_s3r(s3r_uri)
    if bucket in GCS_NATIVE_BUCKETS:
        gcs_bucket, gcs_key = bucket, obj_path
    elif bucket in PBSS_TO_GCS:
        gcs_bucket, prefix = PBSS_TO_GCS[bucket]
        gcs_key = f"{prefix}/{obj_path}" if prefix else obj_path
    else:
        raise ValueError(f"Unknown bucket (no GCS mapping): {bucket}")
    end = start + length - 1
    resp = s3_client.get_object(Bucket=gcs_bucket, Key=gcs_key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


@st.cache_resource(show_spinner="Opening lance dataset...")
def open_dataset(table_uri: str) -> "lance.LanceDataset":
    return lance.dataset(table_uri, storage_options=LANCE_STORAGE_OPTIONS)


def stratified_row_indices(total_rows: int, n: int, seed: int) -> list[int]:
    if total_rows <= n:
        return list(range(total_rows))
    rng = random.Random(seed)
    bucket_size = total_rows / n
    indices = []
    for i in range(n):
        lo = int(i * bucket_size)
        hi = max(lo + 1, min(total_rows, int((i + 1) * bucket_size)))
        indices.append(rng.randrange(lo, hi))
    return indices


@st.cache_data(show_spinner="Loading fragment rows...")
def load_fragment_rows(table_uri: str, fragment_id: int, row_limit: int, source_dataset_filter: str, sample_seed: int):
    dataset = open_dataset(table_uri)
    fragment = dataset.get_fragment(fragment_id)
    total_rows = fragment.count_rows()
    indices = stratified_row_indices(total_rows, row_limit, sample_seed)
    table = fragment.take(indices)
    rows = table.to_pylist()
    if source_dataset_filter:
        rows = [r for r in rows if r.get("source_dataset") == source_dataset_filter]
    return rows, table.schema.names


def main() -> None:
    st.title("🖼️ LanceDB Image Viewer")

    setup_google_auth()

    with st.sidebar:
        st.header("Table")
        table_choice = st.selectbox("Known tables", list(KNOWN_TABLES.keys()) + ["Custom path..."])
        if table_choice == "Custom path...":
            table_uri = st.text_input("gs:// lance path", value="")
        else:
            table_uri = KNOWN_TABLES[table_choice]
            st.caption(table_uri)

        if not table_uri:
            st.stop()

        dataset = open_dataset(table_uri)
        all_fragment_ids = sorted(f.fragment_id for f in dataset.get_fragments())
        st.caption(f"version={dataset.version} · fragments={len(all_fragment_ids)} · rows={dataset.count_rows():,}")

        st.header("Browse")
        fragment_idx = st.number_input(
            "Fragment index (into fragment list, not fragment_id)",
            min_value=0,
            max_value=max(0, len(all_fragment_ids) - 1),
            value=0,
        )
        fragment_id = all_fragment_ids[fragment_idx]
        st.caption(f"fragment_id = {fragment_id}")

        source_dataset_filter = st.text_input("source_dataset filter (optional)", value="")
        row_limit = st.slider("Rows to sample from this fragment", min_value=5, max_value=200, value=30, step=5)

        if "sample_seed" not in st.session_state:
            st.session_state["sample_seed"] = 0
        if st.button("🔀 Resample"):
            st.session_state["sample_seed"] += 1
        st.caption(
            f"Stratified sample: fragment split into {row_limit} equal buckets, "
            f"one random row picked per bucket (seed={st.session_state['sample_seed']})."
        )

    rows, columns = load_fragment_rows(
        table_uri, fragment_id, row_limit, source_dataset_filter, st.session_state["sample_seed"]
    )

    if not rows:
        st.warning("No rows matched in this fragment. Try a different fragment or clear the source_dataset filter.")
        return

    st.subheader(f"Fragment {fragment_id} — {len(rows)} row(s) loaded")

    idx_key = f"row_idx::{table_uri}::{fragment_id}::{source_dataset_filter}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    st.session_state[idx_key] = min(st.session_state[idx_key], len(rows) - 1)

    col_prev, col_next, col_jump = st.columns([1, 1, 3])
    with col_prev:
        if st.button("⬅️ Previous", disabled=st.session_state[idx_key] == 0):
            st.session_state[idx_key] -= 1
    with col_next:
        if st.button("Next ➡️", disabled=st.session_state[idx_key] == len(rows) - 1):
            st.session_state[idx_key] += 1
    with col_jump:
        st.session_state[idx_key] = st.slider("Row", 0, len(rows) - 1, st.session_state[idx_key])

    row = rows[st.session_state[idx_key]]

    img_col, meta_col = st.columns([2, 3])

    with img_col:
        image_range_col = "image_s3_range" if "image_s3_range" in columns else None
        if image_range_col and row.get(image_range_col):
            try:
                s3_client = make_image_s3_client()
                img_bytes = download_image_bytes(s3_client, row[image_range_col])
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        else:
            st.info("No image_s3_range on this row.")

        caption_cols = [c for c in columns if "caption" in c.lower() and row.get(c)]
        for c in caption_cols:
            st.markdown(f"**{c}**")
            st.write(row[c])

    with meta_col:
        st.markdown("**Metadata**")
        skip_cols = set(caption_cols) | ({image_range_col} if image_range_col else set())
        meta_items = {
            k: v for k, v in row.items()
            if k not in skip_cols and not (isinstance(v, list) and len(v) > 16)
        }
        shown = dict(list(meta_items.items())[:MAX_METADATA_COLS_SHOWN])
        st.json(shown, expanded=False)


if __name__ == "__main__":
    main()
