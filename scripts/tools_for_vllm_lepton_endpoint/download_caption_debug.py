"""Download 100 random (uuid, caption, image) triples from captioning_mx_tier1 segments."""
import os
import random
import pathlib
import pandas as pd
import lance
from cosmos_sila.storage import LANCE_STORAGE_OPTIONS
from cosmos_data.gcs_utils import setup_gcs_auth
from cosmos_data.cosmos_downloader import CosmosDownloader

setup_gcs_auth()

OUT = pathlib.Path("/tmp/caption_debug")
OUT.mkdir(exist_ok=True)

TABLE = "gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260508.lance"
SEGMENT_DIR = "/tmp/caption_segments"
pathlib.Path(SEGMENT_DIR).mkdir(exist_ok=True)

# Segment files that are fully written for fragment 4 (segments 0-99, rps=1000)
# We'll pick 1 random segment (1000 rows) and sample 100 from it.
FRAG_ID = 4
random.seed(42)
seg_idx = random.randint(0, 99)
print(f"Using fragment {FRAG_ID}, segment {seg_idx}")

# -- Load caption segment parquet --
import subprocess, sys
seg_gcs = f"gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260508.lance/segments/captioning_mx_tier1/fragment_{FRAG_ID}/rps_1000/segment_{seg_idx}.parquet"
seg_local = f"{SEGMENT_DIR}/frag{FRAG_ID}_seg{seg_idx}.parquet"

if not os.path.exists(seg_local):
    print(f"Downloading segment parquet: {seg_gcs}")
    result = subprocess.run(
        ["aws", "s3", "--profile", "gcs", "cp", seg_gcs.replace("gs://", "s3://"), seg_local],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(result.stderr)
        sys.exit(1)

captions_df = pd.read_parquet(seg_local)
print(f"Segment rows: {len(captions_df)}")

# Sample 100 row indices
sample_indices = sorted(random.sample(range(len(captions_df)), 100))
captions_sample = captions_df.iloc[sample_indices].reset_index(drop=True)

# -- Load uuid + image_s3_range from the main Lance table for the same rows --
# fragment_4 rows in the segment file correspond positionally to rows in the fragment.
# segment_N covers rows [N*rps_1000 .. (N+1)*rps_1000) within the fragment.
rps = 1000
frag_row_start = seg_idx * rps  # offset within fragment

ds = lance.dataset(TABLE, storage_options=LANCE_STORAGE_OPTIONS)
frags = {f.fragment_id: f for f in ds.get_fragments()}
frag = frags[FRAG_ID]

# Read uuid + image_s3_range for just the rows we need
frag_tbl = frag.to_table(columns=["uuid", "image_s3_range"])
frag_df = frag_tbl.to_pandas()

meta_sample = frag_df.iloc[[frag_row_start + i for i in sample_indices]].reset_index(drop=True)
print(f"Meta rows fetched: {len(meta_sample)}")

# -- Download images + write caption files --
downloader = CosmosDownloader(gcs_profile="team-gcs-cosmos")

ok, skipped = 0, 0
for i in range(len(captions_sample)):
    uuid = meta_sample.loc[i, "uuid"]
    caption = captions_sample.loc[i, "captioning_mx_tier1_caption"]
    error = captions_sample.loc[i, "captioning_mx_tier1_error"]
    image_s3_range = meta_sample.loc[i, "image_s3_range"]

    txt_path = OUT / f"{uuid}.txt"
    img_path = OUT / f"{uuid}.jpg"

    # Write caption / error text
    with open(txt_path, "w") as f:
        if caption:
            f.write(caption)
        elif error:
            f.write(f"ERROR: {error}")
        else:
            f.write("(both null)")

    # Download image
    try:
        img_bytes = downloader.download_s3_range(image_s3_range)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        ok += 1
    except Exception as e:
        print(f"  [WARN] {uuid}: image download failed — {e}")
        skipped += 1

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/100 done  (ok={ok}, skip={skipped})")

print(f"\nDone. {ok} images + captions written to {OUT}, {skipped} image failures.")
print("Files:", sorted(OUT.iterdir())[:6], "...")
