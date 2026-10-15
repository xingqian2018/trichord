# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground_v1/interface/run_qwen_image_2p1.py --server.port 8501

import base64
import io
import json
import random
import time
import urllib.request
from pathlib import Path

import streamlit as st
from PIL import Image

RUN = Path("/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/run")
CREDENTIALS = Path.home() / "Project/trichord/credentials/gateway.json"
PE_SYSTEM_PROMPT = (Path(__file__).resolve().parents[1] / "prompt" / "t2i_generation_qwen2p1_prompt_enhancement.md").read_text().strip()
PE = {"dir": "qpe_serve", "model": "Qwen/Qwen-Image-2.1-PE-T2I", "key": "SLURM_QWEN_IMAGE_PE_API_KEY"}
QI = {"dir": "qi_serve", "model": "Qwen/Qwen-Image-2.1", "key": "SLURM_QWEN_IMAGE_API_KEY"}
WH_RATIO_TO_SIZE = {"1:1": "2048x2048", "4:3": "2400x1792", "3:4": "1792x2400", "3:2": "2528x1696", "2:3": "1696x2528", "16:9": "2752x1536", "9:16": "1536x2752"}


def broker(svc: dict) -> str | None:
    f = RUN / svc["dir"] / "broker_url"
    return f.read_text().strip().rstrip("/") if f.is_file() else None


def post(svc: dict, path: str, body: dict) -> dict:
    key = json.loads(CREDENTIALS.read_text())[svc["key"]].strip()
    req = urllib.request.Request(broker(svc) + path, data=json.dumps({"model": svc["model"], **body}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def live(svc: dict) -> int:
    try:
        with urllib.request.urlopen(broker(svc) + "/stats", timeout=5) as r:
            return json.load(r)[svc["model"]]["live_workers"]
    except Exception:
        return 0


st.set_page_config(page_title="Qwen-Image-2.1", layout="wide")
st.title("Qwen-Image-2.1 (PE rewrite → render)")
pe_live, qi_live = live(PE) if broker(PE) else 0, live(QI) if broker(QI) else 0
st.caption(f"PE {'🟢' if pe_live else '🔴'} {broker(PE) or 'not running'} · Qwen-Image {'🟢' if qi_live else '🔴'} {broker(QI) or 'not running'} ({qi_live} GPU servers)")

prompt = st.text_area("Prompt", height=120, placeholder="a corgi playing guitar in the rain")
if st.button("Generate", type="primary", disabled=not (pe_live and qi_live and prompt.strip())):
    t = time.time()
    with st.spinner("PE rewriting ..."):
        r = post(PE, "/v1/chat/completions", {"messages": [{"role": "system", "content": PE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                                              "temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 8192})
        text = (r["choices"][0]["message"].get("content") or "").rpartition("</think>")[2]
        rewrite = json.loads(text[text.find("{"): text.rfind("}") + 1])
    t_pe = time.time() - t
    size = WH_RATIO_TO_SIZE.get(rewrite.get("wh_ratio"), WH_RATIO_TO_SIZE["1:1"])
    seed = random.randrange(2**31)
    with st.spinner(f"Qwen-Image-2.1 {size} ..."):
        r = post(QI, "/v1/images/generations", {"prompt": rewrite["rewritten_prompt"], "size": size, "n": 1,
                                                "num_inference_steps": 40, "true_cfg_scale": 1.0, "seed": seed})
    image = Image.open(io.BytesIO(base64.b64decode(r["data"][0]["b64_json"])))
    left, right = st.columns([3, 2])
    left.image(image, caption=f"{size} ({rewrite.get('wh_ratio')}) · seed {seed} · PE {t_pe:.1f}s + render {time.time() - t - t_pe:.1f}s")
    right.markdown("**Rewritten prompt**")
    right.write(rewrite["rewritten_prompt"])
