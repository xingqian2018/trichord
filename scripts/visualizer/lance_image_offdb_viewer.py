from __future__ import annotations

import json
import os
import random

import boto3
import streamlit as st
from botocore.config import Config

# cd /home/xingqianx/Project/trichord && streamlit run scripts/visualizer/lance_image_offdb_viewer.py

DEFAULT_JSONL = os.path.expanduser("~/data/t2i_data/t2i_cosmos3_kimik3_structured_v0_first50k.jsonl")
DEFAULT_S3_ROOT = "s3://nv-00-10206-vfm/debug/xingqianx/t2i_data/t2i_cosmos3_kimik3_structured_v0"
CREDENTIAL_PATH = os.path.expanduser("~/Project/trichord/credentials/gcs.secret")
IMAGE_NAME = "image.webp"
URL_EXPIRES = 3600

st.set_page_config(page_title="Lance Image OffDB Viewer", page_icon="🖼️", layout="wide")


@st.cache_resource
def s3_client():
    with open(CREDENTIAL_PATH) as f:
        cred = json.load(f)
    return boto3.client(
        "s3",
        endpoint_url=cred["endpoint_url"],
        region_name=cred["region_name"],
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        config=Config(s3={"addressing_style": "path"}, max_pool_connections=32),
    )


@st.cache_data(show_spinner="Loading jsonl...")
def load_rows(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def image_url(s3_root: str, folder_id: str) -> str:
    bucket, _, prefix = s3_root.removeprefix("s3://").partition("/")
    key = f"{prefix.rstrip('/')}/{folder_id}/derived/{IMAGE_NAME}"
    return s3_client().generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=URL_EXPIRES)


def caption_of(row: dict) -> str:
    upsampled = row.get("upsampled_prompt")
    if isinstance(upsampled, dict):
        return upsampled.get("comprehensive_t2i_caption", "")
    return upsampled or ""


def matches(row: dict, query: str, field: str) -> bool:
    if field == "prompt":
        text = row.get("prompt", "")
    elif field == "caption":
        text = caption_of(row)
    else:
        text = json.dumps(row, ensure_ascii=False)
    return query in text.lower()


with st.sidebar:
    st.header("Data")
    jsonl_path = st.text_input("jsonl", DEFAULT_JSONL)
    s3_root = st.text_input("S3 root (sample folders)", DEFAULT_S3_ROOT)
    if not os.path.isfile(jsonl_path):
        st.error(f"not found: {jsonl_path}")
        st.stop()
    rows = load_rows(jsonl_path)
    st.caption(f"{len(rows):,} rows")

    st.header("Filter")
    query = st.text_input("Search", "").strip().lower()
    field = st.radio("Search in", ["prompt", "caption", "everything"], horizontal=True)
    folder = st.text_input("Jump to folder_id", "").strip()
    order = st.radio("Order", ["file order", "shuffled"], horizontal=True)
    seed = st.number_input("Shuffle seed", value=0, step=1, disabled=order != "shuffled")

    st.header("Layout")
    columns = st.slider("Columns", 1, 6, 3)
    per_page = st.select_slider("Per page", [6, 12, 24, 48, 96], value=12)
    show_json = st.checkbox("Show full upsampled_prompt JSON", False)

selected = rows
if folder:
    selected = [r for r in rows if r.get("folder_id") == folder]
elif query:
    selected = [r for r in rows if matches(r, query, field)]
if order == "shuffled" and not folder:
    selected = list(selected)
    random.Random(int(seed)).shuffle(selected)

pages = max(1, (len(selected) + per_page - 1) // per_page)
top = st.columns([3, 1])
top[0].markdown(f"**{len(selected):,}** of {len(rows):,} samples · {pages:,} pages")
page = top[1].number_input("Page", min_value=1, max_value=pages, value=1, step=1)
batch = selected[(page - 1) * per_page: page * per_page]
if not batch:
    st.info("No samples match.")
    st.stop()

images = [image_url(s3_root, r["folder_id"]) for r in batch]

for start in range(0, len(batch), columns):
    cols = st.columns(columns)
    for col, row, image in zip(cols, batch[start:start + columns], images[start:start + columns]):
        with col:
            st.image(image, width="stretch")
            upsampled = row.get("upsampled_prompt")
            ratio = upsampled.get("aspect_ratio", "") if isinstance(upsampled, dict) else ""
            st.caption(f"`{row['folder_id']}` · {str(ratio).replace(',', ':')}")
            st.markdown(f"**Prompt:** {row.get('prompt', '')}")
            with st.expander("comprehensive_t2i_caption"):
                st.write(caption_of(row))
            if show_json:
                with st.expander("upsampled_prompt"):
                    st.json(upsampled, expanded=False)
