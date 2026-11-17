from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import boto3
import lance
import pandas as pd
import streamlit as st
import yaml
from PIL import Image

GCS_CONFIG_PATH = os.path.expanduser("~/.config/dir/config.yaml")
GCS_CONFIG_PROFILE = "team-gcs-cosmos"
IMAGE_CRED_PATH = os.path.expanduser("~/Project/imaginaire4_sila/credentials/gcs.secret")
PROMPTS_PATH = os.path.expanduser("~/tmp/qjudge_final_prompts.json")
DEFAULT_TABLES = [
    "gs://nv-00-10206-vfm/debug/xingqianx/qjudge_sila_smoke/flickr2k.lance",
]
SCORE_COLUMN_HINT = "qjudge"
LANCE_STORAGE_OPTIONS = {
    "client_max_retries": "15",
    "client_retry_timeout": "300",
    "request_timeout": "3m",
    "connect_timeout": "30s",
}
PBSS_TO_GCS = {
    "logged_images": ("nv-00-10206-images", "logged_images"),
    "annotated_data": ("nv-00-10206-images", "annotated_data"),
    "webdataset": ("nv-00-10206-webdataset-images", ""),
    "debug": ("nv-00-10206-images", "debug"),
}
GCS_NATIVE_BUCKETS = {"nv-00-10206-images", "nv-00-10206-webdataset-images"}
GRADES = {0.0: "Fail", 60.0: "Pass", 100.0: "Excel", None: "N/A"}
ICONS = {"Fail": "❌", "Pass": "✅", "Excel": "🌟", "N/A": "·"}
GRADE_ORDER = ["Fail", "Pass", "Excel", "N/A"]
GRID_COLUMNS = 5
IMAGES_PER_PAGE = 50
THUMB_SIDE = 448
CHECKLIST_LINE = re.compile(r"^- ([^:]+): (.+)$")
SORTS = ["random", "most Fail", "most Excel", "fewest Excel", "uuid"]

Image.MAX_IMAGE_PIXELS = None

st.set_page_config(page_title="QJudge Score Check", page_icon="⚖️", layout="wide")


@st.cache_resource
def setup_google_auth() -> None:
    with open(GCS_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    sa_key = json.loads(cfg["gcp"]["service_account_profiles"][GCS_CONFIG_PROFILE]["json_string"])
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(sa_key, tf)
    tf.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name


@lru_cache(maxsize=1)
def image_client():
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
    rest = uri[len("s3r:"):].split("//", 1)[1].rsplit("@", 1)[0]
    bucket_and_path, range_part = rest.rsplit(":", 1)
    bucket, object_path = bucket_and_path.split("/", 1)
    start, length = map(int, range_part.split("-"))
    return bucket, object_path, start, length


@lru_cache(maxsize=64)
def image_bytes(uri: str) -> bytes:
    bucket, path, start, length = parse_s3r(uri)
    if bucket in GCS_NATIVE_BUCKETS:
        key = path
    else:
        bucket, prefix = PBSS_TO_GCS[bucket]
        key = f"{prefix}/{path}" if prefix else path
    resp = image_client().get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{start + length - 1}")
    return resp["Body"].read()


@lru_cache(maxsize=4096)
def thumbnail(uri: str) -> bytes:
    image = Image.open(io.BytesIO(image_bytes(uri)))
    image.draft("RGB", (THUMB_SIDE, THUMB_SIDE))
    image = image.convert("RGB")
    image.thumbnail((THUMB_SIDE, THUMB_SIDE))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=88)
    return out.getvalue()


def safe_thumbnail(uri: str) -> bytes | Exception:
    try:
        return thumbnail(uri)
    except Exception as e:
        return e


@st.cache_data
def load_definitions() -> dict[str, str]:
    if not os.path.exists(PROMPTS_PATH):
        return {}
    definitions = {}
    for request in json.load(open(PROMPTS_PATH)).values():
        for part in request["messages"][1]["content"]:
            for line in part.get("text", "").splitlines():
                match = CHECKLIST_LINE.match(line.strip())
                if match:
                    definitions[match.group(1).strip()] = match.group(2).strip()
    return definitions


@st.cache_resource(show_spinner="Opening table...")
def open_table(uri: str):
    setup_google_auth()
    return lance.dataset(uri, storage_options=LANCE_STORAGE_OPTIONS)


def score_columns(uri: str) -> list[str]:
    names = [f.name for f in open_table(uri).schema]
    hinted = [n for n in names if SCORE_COLUMN_HINT in n]
    return hinted or [n for n in names if n not in ("uuid", "image_s3_range")]


def flatten(scores: dict) -> dict[tuple[str, str, str], float | None]:
    return {(l1, l2, l3): value for l1, l2s in scores.items() for l2, l3s in l2s.items() for l3, value in l3s.items()}


@st.cache_data(show_spinner="Loading scores...")
def load_scores(uri: str, column: str, version: int) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    table = open_table(uri).to_table(columns=["uuid", "image_s3_range", column])
    records = []
    criteria: list[tuple[str, str, str]] = []
    for uuid, image, raw in zip(*(table.column(c).to_pylist() for c in ["uuid", "image_s3_range", column])):
        flat = flatten(json.loads(raw)) if raw else {}
        for key in flat:
            if key not in criteria:
                criteria.append(key)
        records.append({"uuid": uuid, "image_s3_range": image, "parsed": bool(flat), **{key: GRADES.get(value, str(value)) for key, value in flat.items()}})
    frame = pd.DataFrame.from_records(records)
    keys = list(criteria)
    for grade in GRADE_ORDER:
        frame[f"n_{grade}"] = (frame[keys] == grade).sum(axis=1) if keys else 0
    return frame, criteria


def criterion_label(key: tuple[str, str, str], criteria: list[tuple[str, str, str]]) -> str:
    duplicate = sum(1 for c in criteria if c[2] == key[2]) > 1
    return f"{key[0]}/{key[2]}" if duplicate else key[2]


def distribution_table(frame: pd.DataFrame, criteria: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    total = len(frame)
    for key in criteria:
        counts = frame[key].value_counts()
        applicable = total - int(counts.get("N/A", 0))
        row = {"L1": key[0], "L2": key[1], "criterion": key[2]}
        for grade in GRADE_ORDER:
            row[grade] = int(counts.get(grade, 0))
        for grade in ["Fail", "Pass", "Excel"]:
            row[f"{grade} % of applicable"] = round(100 * row[grade] / applicable, 1) if applicable else None
        rows.append(row)
    return pd.DataFrame(rows)


def score_markdown(record: pd.Series, criteria: list[tuple[str, str, str]], highlight: tuple | None) -> str:
    lines = []
    current = None
    for key in criteria:
        if key[0] != current:
            current = key[0]
            lines.append(f"**{current}**")
        grade = record.get(key, "N/A")
        if pd.isna(grade):
            grade = "N/A"
        name = f"**{key[2]}**" if key == highlight else key[2]
        lines.append(f"- {ICONS.get(grade, '?')} {name} · {grade}")
    return "\n".join(lines)


def definitions_table(criteria: list[tuple[str, str, str]], definitions: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{"L1": k[0], "L2": k[1], "criterion": k[2], "judge sees": definitions.get(k[2], "")} for k in criteria])


def select_image(uuid: str) -> None:
    st.session_state["selected_uuid"] = uuid


def clear_selection() -> None:
    st.session_state.pop("selected_uuid", None)


def show_detail(record: pd.Series, criteria: list[tuple[str, str, str]], definitions: dict[str, str]) -> None:
    with st.container(border=True):
        top = st.columns([6, 1])
        top[0].markdown(f"**{record['uuid']}** · ❌ {record['n_Fail']} · ✅ {record['n_Pass']} · 🌟 {record['n_Excel']} · N/A {record['n_N/A']}")
        top[1].button("Close", on_click=clear_selection, type="tertiary", icon=":material/close:")
        image_col, score_col = st.columns([3, 2])
        with image_col:
            try:
                data = image_bytes(record["image_s3_range"])
                with Image.open(io.BytesIO(data)) as image:
                    mime = Image.MIME.get(image.format, "image/jpeg")
                    st.caption(f"{image.width}×{image.height} · {record['image_s3_range']}")
                st.html(f'<img src="data:{mime};base64,{base64.b64encode(data).decode("ascii")}" style="width:100%;height:auto">')
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        with score_col:
            rows = []
            for key in criteria:
                grade = record.get(key, "N/A")
                grade = "N/A" if pd.isna(grade) else grade
                rows.append({"": ICONS.get(grade, "?"), "L1": key[0], "criterion": key[2], "grade": grade, "judge sees": definitions.get(key[2], "")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=640)


def main() -> None:
    st.title("QJudge score check")
    definitions = load_definitions()
    with st.sidebar:
        uri = st.selectbox("Table", DEFAULT_TABLES, accept_new_options=True)
        try:
            columns = score_columns(uri)
        except Exception as e:
            st.error(f"Cannot open table: {e}")
            return
        column = st.selectbox("Score column", columns)
        if st.button("Reload table", icon=":material/refresh:"):
            open_table.clear()
            load_scores.clear()
            st.rerun()
    frame, criteria = load_scores(uri, column, open_table(uri).version)
    labels = {criterion_label(k, criteria): k for k in criteria}
    parsed = int(frame["parsed"].sum())
    st.caption(f"{uri} · column `{column}` · {len(frame):,} rows · {parsed:,} with scores · {len(criteria)} criteria")
    frame = frame[frame["parsed"]]
    if frame.empty:
        st.warning("No scored rows.")
        return

    with st.expander("Grade distribution per criterion", expanded=True):
        st.dataframe(
            distribution_table(frame, criteria),
            hide_index=True,
            width="stretch",
            column_config={f"{g} % of applicable": st.column_config.ProgressColumn(f"{g} %", format="%.1f%%", min_value=0, max_value=100) for g in ["Fail", "Pass", "Excel"]},
        )
        left, right = st.columns(2)
        with left:
            st.markdown("**Images by number of Fail criteria**")
            st.bar_chart(frame["n_Fail"].value_counts().sort_index(), x_label="Fail count", y_label="images")
        with right:
            st.markdown("**Images by number of Excel criteria**")
            st.bar_chart(frame["n_Excel"].value_counts().sort_index(), x_label="Excel count", y_label="images")
        st.caption(f"No Fail on any criterion: {(frame['n_Fail'] == 0).mean():.1%} of scored images")
    with st.expander("Criterion definitions (what the judge sees)"):
        st.dataframe(definitions_table(criteria, definitions), hide_index=True, width="stretch")

    with st.sidebar:
        st.divider()
        criterion_name = st.selectbox("Filter by criterion", ["(any)", *labels])
        grades = st.multiselect("Grade", GRADE_ORDER, default=["Fail"] if criterion_name != "(any)" else [], disabled=criterion_name == "(any)")
        min_fail, max_fail = st.slider("Fail count range", 0, len(criteria), (0, len(criteria)))
        sort = st.selectbox("Order", SORTS)
        seed = st.number_input("Random seed", value=0, step=1, disabled=sort != "random")
        show_scores = st.checkbox("Show per-criterion grades under images", value=True)
    highlight = labels.get(criterion_name)
    view = frame[(frame["n_Fail"] >= min_fail) & (frame["n_Fail"] <= max_fail)]
    if highlight and grades:
        view = view[view[highlight].isin(grades)]
    if sort == "random":
        order = list(range(len(view)))
        random.Random(seed).shuffle(order)
        view = view.iloc[order]
    elif sort == "most Fail":
        view = view.sort_values(["n_Fail", "n_Excel"], ascending=[False, True])
    elif sort == "most Excel":
        view = view.sort_values(["n_Excel", "n_Fail"], ascending=[False, True])
    elif sort == "fewest Excel":
        view = view.sort_values(["n_Excel", "n_Fail"], ascending=[True, False])
    else:
        view = view.sort_values("uuid")

    selected = st.session_state.get("selected_uuid")
    if selected is not None and selected in set(frame["uuid"]):
        show_detail(frame[frame["uuid"] == selected].iloc[0], criteria, definitions)

    pages = max(1, -(-len(view) // IMAGES_PER_PAGE))
    head = st.columns([3, 1])
    head[0].subheader(f"{len(view):,} images" + (f" · {criterion_name} ∈ {', '.join(grades)}" if highlight and grades else ""))
    page = head[1].number_input("Page", 1, pages, 1, key=f"page::{uri}::{criterion_name}::{grades}::{sort}::{min_fail}-{max_fail}") if pages > 1 else 1
    chunk = view.iloc[(page - 1) * IMAGES_PER_PAGE : page * IMAGES_PER_PAGE]
    with ThreadPoolExecutor(max_workers=16) as pool:
        thumbs = list(pool.map(safe_thumbnail, chunk["image_s3_range"]))
    for start in range(0, len(chunk), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (_, record), thumb in zip(cols, chunk.iloc[start : start + GRID_COLUMNS].iterrows(), thumbs[start : start + GRID_COLUMNS]):
            with col:
                if isinstance(thumb, Exception):
                    st.error(f"Failed to load image: {thumb}")
                else:
                    st.image(thumb, width="stretch")
                badge = f"❌ {record['n_Fail']} · 🌟 {record['n_Excel']} · ✅ {record['n_Pass']}"
                if highlight:
                    grade = record.get(highlight)
                    badge = f"{ICONS.get(grade, '?')} {criterion_name}: {grade} · " + badge
                st.button(badge, key=f"open::{record['uuid']}", on_click=select_image, args=(record["uuid"],), type="tertiary")
                if show_scores:
                    with st.expander("grades"):
                        st.markdown(score_markdown(record, criteria, highlight))


main()
