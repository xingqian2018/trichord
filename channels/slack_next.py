import asyncio
import json
import threading
import time
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

CREDS = json.loads((Path(__file__).parent.parent / "credentials" / "slack.json").read_text())

LOG_PATH = Path(__file__).parent / "tmp" / "slack_thread_to_claude_session_mapping.json"
UPDATE_INTERVAL = 1.0  # seconds between chat_update calls (Slack rate limit safety)

SESSION_LOCK = threading.Lock()

app = App(token=CREDS["SLACK_BOT_TOKEN"])


# --- background asyncio loop (slack_bolt handlers are sync+threaded) ---

_loop: asyncio.AbstractEventLoop | None = None


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is not None:
        return _loop
    ready = threading.Event()

    def runner():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        ready.set()
        _loop.run_forever()

    threading.Thread(target=runner, daemon=True).start()
    ready.wait()
    assert _loop is not None
    return _loop


# --- session mapping persistence ---


def load_log() -> dict:
    if not LOG_PATH.exists():
        return {}
    try:
        return json.loads(LOG_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_log(log: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


# --- typed-message rendering ---


def render_message(msg) -> str:
    """Turn one SDK message into a human-readable fragment. Returns '' for events we don't surface."""
    if isinstance(msg, AssistantMessage):
        parts = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                parts.append(f"\n> :wrench: `{block.name}` {json.dumps(block.input)[:120]}\n")
            elif isinstance(block, ThinkingBlock):
                pass
        return "".join(parts)
    if isinstance(msg, UserMessage):
        content = msg.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, ToolResultBlock):
                    result = block.content
                    if isinstance(result, list):
                        result = " ".join(c.get("text", "") for c in result if isinstance(c, dict))
                    snippet = str(result or "").strip().replace("\n", " ")[:160]
                    return f"> _result:_ {snippet}\n"
    return ""


def extract_init(msg) -> tuple[str | None, str | None]:
    """If msg is the init SystemMessage, return (session_id, model); else (None, None)."""
    if isinstance(msg, SystemMessage):
        data = getattr(msg, "data", {}) or {}
        return data.get("session_id"), data.get("model")
    return None, None


# --- core async driver ---


async def run_claude(text: str, session_id: str | None, on_event) -> None:
    options = ClaudeAgentOptions(resume=session_id)
    async for msg in query(prompt=text, options=options):
        on_event(msg)


def drive_claude(
    text: str,
    session_id: str | None,
    channel: str,
    msg_ts: str,
    logger,
) -> tuple[str, str | None, str | None]:
    """Block the slack handler thread while claude streams back into `msg_ts`.
    Returns (final_text, session_id_from_init, model_from_init)."""
    accumulated = ""
    last_update = 0.0
    init_sid: str | None = None
    init_model: str | None = None

    def push(body: str):
        try:
            app.client.chat_update(channel=channel, ts=msg_ts, text=body)
        except Exception as e:
            logger.warning(f"chat_update failed: {e}")

    def on_event(msg):
        nonlocal accumulated, last_update, init_sid, init_model
        sid, model = extract_init(msg)
        if sid and init_sid is None:
            init_sid = sid
            init_model = model
            return
        if isinstance(msg, ResultMessage):
            return
        fragment = render_message(msg)
        if not fragment:
            return
        accumulated += fragment
        now = time.time()
        if now - last_update > UPDATE_INTERVAL:
            last_update = now
            push(accumulated)

    loop = _ensure_loop()
    fut = asyncio.run_coroutine_threadsafe(run_claude(text, session_id, on_event), loop)
    try:
        fut.result()
    except Exception as e:
        logger.exception("claude query failed")
        push((accumulated or "") + f"\n:x: `{e}`")
        raise

    push(accumulated or "_(no response)_")
    return accumulated, init_sid, init_model


# --- slack handler ---


@app.event("message")
def handle_dm_events(event, say, logger):
    if event.get("channel_type") != "im":
        return
    if event.get("bot_id"):
        return

    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    logger.info(f"DM from user={user} channel={channel}: {text[:30]}")
    app.client.reactions_add(channel=channel, timestamp=event["ts"], name="eyes")

    key = thread_key(channel, thread_ts)
    log = load_log()
    session_id = log.get(key)

    placeholder = say(text=":hourglass_flowing_sand: _thinking..._", thread_ts=thread_ts)
    msg_channel = placeholder["channel"]
    msg_ts = placeholder["ts"]

    if session_id:
        logger.info(f"Resuming claude session {session_id} for {key}")
    else:
        logger.info(f"Starting new claude session for {key}")

    reply, init_sid, init_model = drive_claude(text, session_id, msg_channel, msg_ts, logger)

    if init_sid and init_sid != session_id:
        with SESSION_LOCK:
            log = load_log()
            log[key] = init_sid
            save_log(log)
        logger.info(f"Saved session {init_sid} for {key}")
        if not session_id:
            say(text=f":gear: `{init_model or '?'}` / `{init_sid}`", thread_ts=thread_ts)

    logger.info(f"Claude response: {reply[:30]}")


if __name__ == "__main__":
    _ensure_loop()
    SocketModeHandler(app, CREDS["SLACK_APP_TOKEN"]).start()
