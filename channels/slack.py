import json
import subprocess
import threading
import time
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

CREDS = json.loads((Path(__file__).parent.parent / "credentials" / "slack.json").read_text())

LOG_PATH = Path(__file__).parent / "tmp" / "slack_thread_to_claude_session_mapping.json"
UPDATE_INTERVAL = 1.0  # seconds between chat_update calls (Slack rate limit safety)

BOT_USER_ID = None
SESSION_LOCK = threading.Lock()

app = App(token=CREDS["SLACK_BOT_TOKEN"])


def get_bot_user_id():
    global BOT_USER_ID
    if BOT_USER_ID is None:
        BOT_USER_ID = app.client.auth_test()["user_id"]
    return BOT_USER_ID


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


def spawn_claude_with_init(text: str, session_id: str | None = None) -> tuple[subprocess.Popen, dict]:
    """Spawn claude in stream-json mode; block only on reading the first event (the `system`/init
    event, which carries session_id + model). Returns (still-streaming proc, init_event)."""
    args = ["claude", "-p", text, "--output-format", "stream-json", "--verbose"]
    if session_id:
        args += ["--resume", session_id]
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    first = proc.stdout.readline()
    if not first:
        proc.wait()
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"claude exited before init event: {err.strip()}")
    try:
        init = json.loads(first)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"unexpected first line: {first[:200]!r}") from e
    return proc, init


def render_event(evt: dict) -> str:
    """Turn one stream-json event into a human-readable fragment to append to the live message.
    Returns '' for events we don't surface."""
    etype = evt.get("type")
    if etype == "assistant":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tname = block.get("name", "?")
                tinput = block.get("input", {})
                parts.append(f"\n> :wrench: `{tname}` {json.dumps(tinput)[:120]}\n")
            elif btype == "thinking":
                parts.append("")
        return "".join(parts)
    if etype == "user":
        for block in evt.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                content = block.get("content", "")
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                snippet = str(content).strip().replace("\n", " ")[:160]
                return f"> _result:_ {snippet}\n"
        return ""
    if etype in ("permission_request", "permission_denied", "permission"):
        return f"\n:warning: **permission**: `{json.dumps(evt)[:300]}`\n"
    return ""


def stream_to_slack(proc: subprocess.Popen, channel: str, msg_ts: str, header: str, logger) -> str:
    """Read stream-json events line by line and live-update the Slack message. Returns final text."""
    accumulated = ""
    last_update = 0.0

    def push(text: str):
        try:
            app.client.chat_update(channel=channel, ts=msg_ts, text=text)
        except Exception as e:
            logger.warning(f"chat_update failed: {e}")

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"non-json stream line: {line[:120]!r}")
            continue

        fragment = render_event(evt)
        if fragment:
            accumulated += fragment

        now = time.time()
        if accumulated and now - last_update > UPDATE_INTERVAL:
            push(header + accumulated)
            last_update = now

    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"claude failed: {err.strip()}")

    push(header + (accumulated or "_(no response)_"))
    return accumulated


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
        proc, init = spawn_claude_with_init(text, session_id=session_id)
    else:
        logger.info(f"Starting new claude session for {key}")
        with SESSION_LOCK:
            proc, init = spawn_claude_with_init(text)
            new_sid = init.get("session_id")
            if not new_sid:
                proc.kill()
                raise RuntimeError(f"init event missing session_id: {init}")
            session_id = new_sid
            log = load_log()
            log[key] = session_id
            save_log(log)
        logger.info(f"Saved session {session_id} for {key}")
        model = init.get("model", "?")
        say(text=f":gear: `{model}` / `{session_id}`", thread_ts=thread_ts)

    reply = stream_to_slack(proc, msg_channel, msg_ts, "", logger)
    logger.info(f"Claude response: {reply[:30]}")


if __name__ == "__main__":
    SocketModeHandler(app, CREDS["SLACK_APP_TOKEN"]).start()
