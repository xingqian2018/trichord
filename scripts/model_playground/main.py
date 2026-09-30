# Run this
# streamlit run /home/xingqianx/Project/trichord/scripts/model_playground/main.py

import base64
import json
import difflib
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from taxonomy.helper import Taxonomy  # noqa: E402
from tool import TOOL_REGISTRY  # noqa: E402
from tool.agent import Agent  # noqa: E402
from utils import (  # noqa: E402
    MODEL_CHOICE,
    RENDER_TARGET_AREA,
    SCRATCH_ROOT,
    infer_canvas_size,
    inject_reset_css,
    inline_local_images,
    render_html_to_png,
    resolve_model_string,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MODEL = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parent / "prompt"
RENDERED_POSTER_DIR = SCRATCH_ROOT / "rendered_posters"
REASONING_EFFORTS = ["default", "low", "high", "max"]
TAXONOMY_NONE = "(none)"
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


def execute_tool_call(tool_instances: dict, name: str, arguments_json: str) -> dict:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    if name not in tool_instances:
        tool_instances[name] = TOOL_REGISTRY[name]()
    args = json.loads(arguments_json) if arguments_json else {}
    return tool_instances[name].run(args)


def run_turn(agent: Agent, tool_instances: dict, status: dict, user_input: str | None, images: list | None):
    def set_stage(text: str):
        status["stage"] = text
        status["since"] = time.time()

    try:
        for call in agent.pending_tool_calls():
            note = f"Tool call '{call['name']}' was interrupted before a result was recorded."
            agent.add_tool_result(call["id"], {"text": note, "images": []})
        if user_input is not None:
            agent.add_user_message(user_input, images=images or None)
        step_no = 0
        while True:
            step_no += 1
            set_stage(f"model step {step_no} ({agent.model_name})")
            agent.step()
            pending = agent.pending_tool_calls()
            if not pending:
                return
            for call in pending:
                set_stage(f"tool: {call['name']}")
                try:
                    tool_result = execute_tool_call(tool_instances, call["name"], call["arguments"])
                except Exception as e:
                    tool_result = {"text": f"Tool call '{call['name']}' failed: {type(e).__name__}: {e}", "images": []}
                agent.add_tool_result(call["id"], tool_result)
    except Exception as e:
        for call in agent.pending_tool_calls():
            agent.add_tool_result(call["id"], {"text": f"Run aborted: {type(e).__name__}: {e}", "images": []})
        agent.append({"role": "assistant", "content": f"Run failed: {type(e).__name__}: {e}"}, {"finish_reason": "error"})


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

    def _sync_tools_json():
        selected = [TOOL_REGISTRY[n].SCHEMA for n in TOOL_REGISTRY if st.session_state.get(f"use_tool_{n}")]
        st.session_state.tools_json_text = json.dumps(selected, indent=2, ensure_ascii=False) if selected else ""

    st.markdown("**Tools**")
    for tool_name in TOOL_REGISTRY:
        st.checkbox(tool_name, key=f"use_tool_{tool_name}", on_change=_sync_tools_json)
    if "tools_json_text" not in st.session_state:
        st.session_state.tools_json_text = ""
    tools_json_text = st.text_area(
        "Tool definitions (JSON list, OpenAI function-calling format)",
        key="tools_json_text",
        height=200,
    )
    tools_def = None
    if tools_json_text.strip():
        try:
            tools_def = json.loads(tools_json_text)
        except json.JSONDecodeError as e:
            st.error(f"Tool definitions JSON is invalid: {e}")

    if st.button("Clear chat"):
        st.session_state.pop("agent", None)
        st.session_state.pop("worker", None)
        for key in list(st.session_state.keys()):
            if key.startswith(("html_preview_", "html_render_", "use_rendered_")):
                del st.session_state[key]
        st.session_state.pending_attachment = None
        st.rerun()


def new_agent() -> Agent:
    return Agent(
        model_name,
        system_prompt,
        tools_def or [],
        max_tokens=int(max_tokens),
        temperature=temperature if send_temperature else None,
        reasoning_effort=None if reasoning_effort == "default" else reasoning_effort,
    )


if "agent" not in st.session_state:
    st.session_state.agent = new_agent()
if "pending_attachment" not in st.session_state:
    st.session_state.pending_attachment = None
if "tool_instances" not in st.session_state:
    st.session_state.tool_instances = {}
if "run_status" not in st.session_state:
    st.session_state.run_status = {}

agent: Agent = st.session_state.agent
agent.set_model(model_name)
agent.set_system_prompt(system_prompt)
agent.set_tools(tools_def or [])
agent.max_tokens = int(max_tokens)
agent.temperature = temperature if send_temperature else None
agent.reasoning_effort = None if reasoning_effort == "default" else reasoning_effort

st.caption(f"Resolved model string: `{resolve_model_string(model_name)}` · session `{agent.session_id}`")


def render_reasoning(meta: dict):
    if meta.get("reasoning"):
        with st.expander("🧠 Thinking trace"):
            st.markdown(meta["reasoning"])


def render_tool_calls(meta: dict):
    if meta.get("tool_calls"):
        try:
            pretty = json.dumps(json.loads(meta["tool_calls"]), indent=4, ensure_ascii=False)
        except (TypeError, ValueError):
            pretty = meta["tool_calls"]
        with st.expander("🔧 Tool calls"):
            st.code(pretty, language="json")


def render_tool_images(meta: dict):
    for path in meta.get("images") or []:
        st.image(path)


def render_finish_reason(meta: dict):
    if "finish_reason" in meta:
        st.caption(f"finish_reason: {meta['finish_reason']}")


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


def render_html_preview_buttons(text: str, msg_idx: int, meta: dict):
    if not isinstance(text, str):
        return
    for block_idx, html_code in enumerate(extract_html_blocks(text)):
        toggle_key = f"html_preview_{msg_idx}_{block_idx}"
        if toggle_key not in st.session_state:
            st.session_state[toggle_key] = False

        rendered_path = meta.get("rendered_html_path")
        if rendered_path and Path(rendered_path).is_file():
            rendered_html = Path(rendered_path).read_text()
            if normalize_html(rendered_html) != normalize_html(html_code):
                st.warning(
                    f"Delivered HTML differs from the last rendered version ({Path(rendered_path).name}) — "
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
                    out = RENDERED_POSTER_DIR / f"ui_{agent.session_id}_{msg_idx}_{block_idx}.png"
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


def render_message(idx: int, msg: dict, meta: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_reasoning(meta)
        render_content(msg["content"])
        if msg["role"] == "assistant":
            if meta.get("finish_reason") == "stop":
                render_html_preview_buttons(msg["content"], idx, meta)
            render_tool_calls(meta)
            render_finish_reason(meta)
        elif msg["role"] == "tool":
            render_tool_images(meta)
            st.caption(f"tool_call_id: {msg['tool_call_id']}")


render_baseline = len(agent.messages)
for idx in range(render_baseline):
    render_message(idx, agent.messages[idx], agent.meta[idx])

def is_running() -> bool:
    worker = st.session_state.get("worker")
    return worker is not None and worker.is_alive()


def start_worker(user_input: str | None, images: list | None):
    worker = threading.Thread(
        target=run_turn,
        args=(agent, st.session_state.tool_instances, st.session_state.run_status, user_input, images),
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
        render_message(idx, agent.messages[idx], agent.meta[idx])
    status = st.session_state.run_status
    elapsed = int(time.time() - status.get("since", time.time()))
    st.status(f"thinking... {status.get('stage', agent.model_name)} · {elapsed}s", state="running", expanded=False)


live_tail()

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

