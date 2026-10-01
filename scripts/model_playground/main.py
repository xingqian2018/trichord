# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground/main.py

import base64
import json
import difflib
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from taxonomy.helper import Taxonomy  # noqa: E402
from agent.base import Agent  # noqa: E402
from agent.poster_generation import poster_generation  # noqa: E402
from utils import MODEL_CHOICE, image_conversion, resolve_model_string  # noqa: E402
from tool.render_poster import RENDER_TARGET_AREA, infer_canvas_size, inject_reset_css, inline_local_images, render_html_to_png  # noqa: E402

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
RENDERED_POSTER_DIR = Path("/tmp/model_playground/rendered_posters")
EMPTY_REPLY = "*(empty response — check the terminal log for API errors)*"
REASONING_EFFORTS = ["default", "low", "high", "max"]
TAXONOMY_NONE = "(none)"
RESULT_ROOT = Path.home() / "agentic_result"
TOPIC_MESSAGE_TEMPLATE = "Please generate a poster with the following topic:\n{topic}"

HTML_BLOCK_RE = re.compile(r"```\s*html\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
GENERIC_FENCE_RE = re.compile(r"```(\w*)\s*\n?(.*?)```", re.DOTALL)
HTML_DOC_RE = re.compile(r"<!doctype html|<html[\s>]", re.IGNORECASE)


def list_prompt_presets() -> dict[str, Path]:
    if not PROMPT_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(PROMPT_DIR.glob("*.md"))}


@st.cache_resource
def get_taxonomy() -> Taxonomy:
    return Taxonomy()


def run_turn(agent: Agent, user_input: str | None, images: list | None):
    image_urls = [image_conversion(data, dst_fmt="data_url") for data in images or []]
    agent.run({"prompt": user_input, "image_urls": image_urls})


def extract_html_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in HTML_BLOCK_RE.findall(text)]
    if blocks:
        return blocks
    for lang, body in GENERIC_FENCE_RE.findall(text):
        if lang.lower() != "html" and HTML_DOC_RE.search(body):
            blocks.append(body.strip())
    return blocks


def scaled_size(width: int, height: int, target_area: int) -> tuple[int, int]:
    scale = (target_area / (width * height)) ** 0.5
    return max(1, round(width * scale)), max(1, round(height * scale))


def normalize_html(html_code: str) -> str:
    return re.sub(r"\s+", " ", html_code).strip()


st.set_page_config(page_title="Model Playground", layout="wide")
st.title("Model Playground")

with st.sidebar:
    model_options = list(MODEL_CHOICE.keys())
    model_name = st.selectbox("Model", model_options, index=model_options.index(DEFAULT_MODEL))
    # gpt-6-astra rejects the temperature parameter, so it is opt-in.
    send_temperature = st.checkbox("Send temperature", value=False)
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.05, disabled=not send_temperature)
    max_tokens = st.number_input("Max tokens", min_value=256, max_value=131072, value=16384, step=1024)
    reasoning_effort = st.selectbox("Reasoning effort (kimi)", REASONING_EFFORTS, index=0)

    prompt_presets = list_prompt_presets()

    def _load_preset():
        selected = st.session_state.preset_select
        if selected == "Custom":
            st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        else:
            st.session_state.system_prompt = prompt_presets[selected].read_text()

    st.radio(
        "System prompt preset",
        ["Custom", *prompt_presets.keys()],
        key="preset_select",
        on_change=_load_preset,
    )
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    system_prompt = st.text_area("System prompt", key="system_prompt", height=150)

    def _load_topic():
        selected = st.session_state.taxonomy_select
        if selected == TAXONOMY_NONE:
            st.session_state.show_draft = False
            return
        st.session_state.show_draft = True
        st.session_state.draft_editor = TOPIC_MESSAGE_TEMPLATE.format(topic=selected)

    st.selectbox(
        "Taxonomy topic",
        [TAXONOMY_NONE, *get_taxonomy().list_topics()],
        key="taxonomy_select",
        on_change=_load_topic,
    )

    if st.button("Clear chat"):
        st.session_state.pop("agent", None)
        st.session_state.pop("worker", None)
        for key in list(st.session_state.keys()):
            if key.startswith(("html_preview_", "html_render_", "use_rendered_")):
                del st.session_state[key]
        st.session_state.pending_attachment = None
        st.rerun()


def new_agent() -> Agent:
    return poster_generation(
        "PosterGenerationAgent_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
        model_name,
        system_prompt,
        max_tokens=int(max_tokens),
        temperature=temperature if send_temperature else None,
        reasoning_effort=None if reasoning_effort == "default" else reasoning_effort,
    )


def agent_matches_sidebar(candidate: Agent) -> bool:
    return (
        candidate.model_name == model_name
        and candidate.system_prompt == system_prompt
        and candidate.max_tokens == int(max_tokens)
        and candidate.temperature == (temperature if send_temperature else None)
        and candidate.reasoning_effort == (None if reasoning_effort == "default" else reasoning_effort)
    )


if "agent" not in st.session_state:
    st.session_state.agent = new_agent()
elif not st.session_state.agent.messages and not agent_matches_sidebar(st.session_state.agent):
    st.session_state.agent = new_agent()
if "pending_attachment" not in st.session_state:
    st.session_state.pending_attachment = None

agent: Agent = st.session_state.agent
if agent.messages and not agent_matches_sidebar(agent):
    st.sidebar.caption("Sidebar changes apply after Clear chat.")
agent.max_tokens = int(max_tokens)
agent.temperature = temperature if send_temperature else None
agent.reasoning_effort = None if reasoning_effort == "default" else reasoning_effort

with st.sidebar:
    st.markdown("**Tools** (configured by PosterGenerationAgent)")
    for tool_name in agent.tool_registry:
        st.caption(f"• tool: {tool_name}")
    for declared in agent.agents:
        st.caption(f"• agent: {declared['agent']['type']} ({declared['agent']['name']})")
    with st.expander("Tool definitions"):
        st.code(json.dumps([*agent.tools, *agent.agents], indent=2, ensure_ascii=False), language="json")

st.caption(f"Resolved model string: `{resolve_model_string(model_name)}` · agent `{agent.agent_name}`")


def render_reasoning(msg: dict):
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        with st.expander("🧠 Thinking trace"):
            st.markdown(reasoning)


def render_tool_calls(msg: dict):
    if msg.get("tool_calls"):
        with st.expander("🔧 Tool calls"):
            st.code(json.dumps(msg["tool_calls"], indent=4, ensure_ascii=False), language="json")


def render_tool_images(msg: dict):
    handler = agent.tool_instances.get(msg["tool_call_id"])
    for path in getattr(handler, "images", None) or []:
        st.image(path)


def render_content(content):
    if not content:
        return
    if isinstance(content, str):
        st.markdown(content)
        return
    for block in content:
        if block["type"] == "text":
            st.markdown(block["text"])
        elif block["type"] == "image_url":
            _, encoded = block["image_url"]["url"].split(",", 1)
            st.image(base64.b64decode(encoded))


def html_diff(rendered_html: str, delivered_html: str, max_lines: int = 120) -> str:
    a = [line.strip() for line in rendered_html.splitlines() if line.strip()]
    b = [line.strip() for line in delivered_html.splitlines() if line.strip()]
    diff = list(difflib.unified_diff(a, b, fromfile="rendered (reviewed)", tofile="delivered", lineterm="", n=1))
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... ({len(diff) - max_lines} more lines)"]
    return "\n".join(diff)


def render_html_preview_buttons(text: str, msg_idx: int, msg: dict):
    if not isinstance(text, str):
        return
    for block_idx, html_code in enumerate(extract_html_blocks(text)):
        toggle_key = f"html_preview_{msg_idx}_{block_idx}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False

        rendered_html = getattr(agent, "cached_html", None)
        if rendered_html:
            if normalize_html(rendered_html) != normalize_html(html_code):
                st.warning(
                    "Delivered HTML differs from the last rendered version — "
                    "the critic reviewed that one, not this block."
                )
                with st.expander("Show diff (rendered → delivered)"):
                    st.code(html_diff(rendered_html, html_code) or "(only whitespace/line-structure differences)", language="diff")
                if st.checkbox("Use the rendered version for Preview / Render", key=f"use_rendered_{msg_idx}_{block_idx}"):
                    html_code = rendered_html

        canvas_width, canvas_height = infer_canvas_size(html_code)
        preview_width, preview_height = scaled_size(canvas_width, canvas_height, RENDER_TARGET_AREA)

        preview_col, render_col = st.columns(2)
        with preview_col:
            if st.button("Preview", key=f"btn_{toggle_key}"):
                st.session_state[toggle_key] = not st.session_state[toggle_key]
        with render_col:
            render_clicked = st.button("Render", key=f"btn_render_{msg_idx}_{block_idx}")

        if st.session_state[toggle_key]:
            preview_html = inject_reset_css(inline_local_images(html_code), canvas_width, canvas_height)
            components.html(preview_html, width=preview_width, height=preview_height, scrolling=False)

        png_key = f"html_render_{msg_idx}_{block_idx}"
        if render_clicked:
            with st.spinner("Rendering..."):
                try:
                    out = RENDERED_POSTER_DIR / f"ui_{agent.agent_name}_{msg_idx}_{block_idx}.png"
                    st.session_state[png_key] = str(render_html_to_png(html_code, out, RENDER_TARGET_AREA))
                except Exception as e:
                    st.error(f"Render failed: {e}")

        if png_key in st.session_state:
            png_path = Path(st.session_state[png_key])
            st.image(str(png_path), caption=png_path.name)
            dl_col, attach_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "Download PNG",
                    data=png_path.read_bytes(),
                    file_name=png_path.name,
                    mime="image/png",
                    key=f"dl_{png_key}",
                )
            with attach_col:
                if st.button("Add as attachment", key=f"attach_{png_key}"):
                    st.session_state.pending_attachment = png_path.read_bytes()
                    st.rerun()


def render_message(idx: int, msg: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_reasoning(msg)
        render_content(msg["content"])
        if msg["role"] == "assistant" and not msg["content"] and not msg.get("tool_calls"):
            st.markdown(EMPTY_REPLY)
        if msg["role"] == "assistant":
            if not msg.get("tool_calls"):
                render_html_preview_buttons(msg["content"], idx, msg)
            render_tool_calls(msg)
        elif msg["role"] == "tool":
            render_tool_images(msg)
            st.caption(f"tool_call_id: {msg['tool_call_id']}")


render_baseline = len(agent.messages)
for idx in range(render_baseline):
    render_message(idx, agent.messages_extended[idx])

def is_running() -> bool:
    worker = st.session_state.get("worker")
    return worker is not None and worker.is_alive()


def start_worker(user_input: str | None, images: list | None):
    worker = threading.Thread(
        target=run_turn,
        args=(agent, user_input, images),
        daemon=True,
    )
    st.session_state.worker = worker
    worker.start()


running = is_running()


@st.fragment(run_every=0.7)
def live_tail():
    if not is_running():
        if st.session_state.get("was_running"):
            st.session_state.was_running = False
            st.rerun(scope="app")
        return
    st.session_state.was_running = True
    for idx in range(render_baseline, len(agent.messages)):
        render_message(idx, agent.messages_extended[idx])
    st.status("thinking...", state="running", expanded=False)


live_tail()

if not running and agent.messages:
    default_dir = str(RESULT_ROOT / agent.agent_name)
    if st.session_state.get("save_dir_session") != agent.agent_name:
        st.session_state.save_dir = default_dir
        st.session_state.save_dir_session = agent.agent_name
    path_col, btn_col = st.columns([5, 1])
    with path_col:
        st.text_input("Save path", key="save_dir", label_visibility="collapsed")
    with btn_col:
        if st.button("Save result", use_container_width=True):
            try:
                agent.save_history(Path(st.session_state.save_dir).expanduser())
                saved = Path(st.session_state.save_dir).expanduser()
                st.success(f"Saved to {saved}")
            except Exception as e:
                st.error(f"Save failed: {type(e).__name__}: {e}")

if st.session_state.pending_attachment is not None:
    cap_col, remove_col = st.columns([5, 1])
    with cap_col:
        st.image(st.session_state.pending_attachment, width=120, caption="Rendered PNG attached to next message")
    with remove_col:
        if st.button("Remove"):
            st.session_state.pending_attachment = None
            st.rerun()

if st.session_state.get("show_draft"):
    st.text_area("Draft message (from taxonomy topic)", key="draft_editor", height=100)
    send_col, discard_col = st.columns([1, 5])
    with send_col:
        if st.button("Send draft", disabled=running):
            st.session_state.submit_draft = st.session_state.draft_editor
            st.session_state.show_draft = False
            st.rerun()
    with discard_col:
        if st.button("Discard draft"):
            st.session_state.show_draft = False
            st.rerun()

chat_value = st.chat_input(
    "Ask anything...", accept_file="multiple", file_type=["png", "jpg", "jpeg", "webp", "gif"], disabled=running
)
draft = st.session_state.pop("submit_draft", None)
user_input, attached = None, []
if chat_value is not None and (chat_value.text or chat_value.files):
    user_input = chat_value.text or ""
    attached = [f.getvalue() for f in chat_value.files]
elif draft:
    user_input = draft

if user_input is not None and not running:
    images = attached
    if not images and st.session_state.pending_attachment is not None:
        images.append(st.session_state.pending_attachment)
    st.session_state.pending_attachment = None
    start_worker(user_input, images)
    st.rerun()

if not running and agent.pending_tool_calls():
    start_worker(None, None)
    st.rerun()

