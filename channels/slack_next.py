import asyncio
import json
from collections import deque
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

CREDS = json.loads((Path(__file__).parent.parent / "credentials" / "slack.json").read_text())

LOG_PATH = Path(__file__).parent / "tmp" / "slack_thread_to_claude_session_mapping.json"
SLACK_MAX_TEXT = 38000  # Slack chat.update hard limit is 40000 chars

YES = {"yes", "y", "allow", "approve", "ok", "sure"}
NO = {"no", "n", "deny", "reject", "stop"}
YES_ALL = {"yes all", "yesall", "allow all"}
NO_ALL = {"no all", "noall", "deny all"}

app = AsyncApp(token=CREDS["SLACK_BOT_TOKEN"])

PENDING: dict[str, deque[asyncio.Future]] = {}

AUTO_APPROVE: dict[str, bool] = {}


def thread_key(channel: str, thread_ts: str) -> str:
    return f"{channel}:{thread_ts}"


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


def render_message(msg) -> str:
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


async def try_finalize_gear_ts(channel, gear_ts, msg):
    if not isinstance(msg, SystemMessage):
        return gear_ts
    sid = msg.data.get('session_id')
    model = msg.data.get('model')
    if not sid:
        return gear_ts
    await app.client.chat_update(
        channel=channel,
        ts=gear_ts,
        text=f":gear: `{model or '?'}` / `{sid}`",
    )
    return None


async def commit(channel, thread_ts, body: str) -> str:
    if len(body) > SLACK_MAX_TEXT:
        body = "_(...earlier output truncated...)_\n\n" + body[-SLACK_MAX_TEXT:]

    if thread_ts is None:
        posted = await app.client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text=body
        )
        return posted["ts"]
    await app.client.chat_update(channel=channel, ts=thread_ts, text=body)
    return thread_ts


async def drive_claude(
    text: str,
    session_id: str | None,
    channel: str,
    thread_ts: str,
    header_ts, str,
    logger,
) -> tuple[str, str | None]:
    
    pending_permission = []

    async def can_use_tool(tool_name, tool_input, _context):

        fut: asyncio.Future[tuple[str, str]] = loop.create_future()
        pending_permission.setdefault(key, deque()).append(fut)

        preview = json.dumps(tool_input)[:400]
        prompt_ts: str | None = None
        try:
            prompt = await app.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=(
                    f":lock: allow `{tool_name}`?\n"
                    f"```{preview}```\n"
                    f"reply *yes* / *no* / *yes all* / *no all* "
                    f"— any other text interrupts claude with your message"
                ),
            )
            prompt_ts = prompt["ts"]
        except Exception:
            logger.exception("failed to post permission prompt")

        try:
            verdict, message = await fut
        except asyncio.CancelledError:
            return PermissionResultDeny(message="permission cancelled")

        if prompt_ts is not None:
            verdict_label = {
                "allow": ":white_check_mark: _allowed_",
                "deny": ":x: _denied_",
                "interrupt": ":arrow_right_hook: _interrupted_",
            }.get(verdict, f"_{verdict}_")
            try:
                await app.client.chat_update(
                    channel=channel,
                    ts=prompt_ts,
                    text=f":lock: `{tool_name}` — {verdict_label}\n```{preview}```",
                )
            except Exception as e:
                logger.warning(f"update permission prompt failed: {e}")

        if verdict == "allow":
            return PermissionResultAllow(updated_input=tool_input)
        if verdict == "interrupt":
            return PermissionResultDeny(message=message, interrupt=True)
        return PermissionResultDeny(message=message or "user denied")

    try:
        main_msg = await app.client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=":hourglass_flowing_sand: _thinking..._",
        )
        main_ts = main_msg["ts"]
    except Exception as e:
        main_ts = None

    options = ClaudeAgentOptions(resume=session_id, can_use_tool=can_use_tool)

    key = thread_key(channel, thread_ts)
    loop = asyncio.get_running_loop()

    #############
    # main loop #
    #############

    async with ClaudeSDKClient(options=options) as client:
        response_str = ""
        query_str = text

        await client.query(query_str)
        async for msg in client.receive_response():
            if header_ts is not None:
                header_ts = await try_finalize_gear_ts(channel, header_ts, msg)
            if isinstance(msg, ResultMessage):
                if response_str:
                    main_ts = await commit(channel, main_ts, response_str)
                break
            fragment = render_message(msg)
            if fragment:
                response_str += fragment

    return None


@app.event("message")
async def handle_dm_events(event, say, logger):
    if event.get("channel_type") != "im":
        return
    if event.get("subtype") in ("message_changed", "message_deleted", "bot_message"):
        return
    if event.get("bot_id"):
        return

    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")
    is_new_session = event.get("thread_ts") is None
    thread_ts = event.get("thread_ts") or event.get("ts")
    key = thread_key(channel, thread_ts)
    stripped = text.strip().lower()

    queue = PENDING.get(key)
    if queue:
        is_yes_all = stripped in YES_ALL
        is_no_all = stripped in NO_ALL
        is_yes = stripped in YES
        is_no = stripped in NO

        if is_yes_all:
            AUTO_APPROVE[key] = True
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    fut.set_result(("allow", ""))
            return

        if is_no_all:
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    fut.set_result(("deny", "user denied"))
            return

        if is_yes or is_no:
            result = ("allow", "") if is_yes else ("deny", "user denied")
            while queue:
                fut = queue.popleft()
                if not fut.done():
                    fut.set_result(result)
                    break
            return

        while queue:
            fut = queue.popleft()
            if not fut.done():
                fut.set_result(("interrupt", text))
        return

    logger.info(f"DM from user={user} channel={channel}: {text[:30]}")

    session_id = None if is_new_session else load_log().get(key)

    header_ts = None
    try:
        if is_new_session:
            header_msg = await app.client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=":gear: _starting new session..._"
            )
            header_ts = header_msg['ts']
    except:
        header_ts = None

    reply, init_sid = await drive_claude(
        text, session_id, channel, thread_ts, header_ts, logger
    )

    if init_sid and init_sid != session_id:
        log = load_log()
        log[key] = init_sid
        save_log(log)
        logger.info(f"Saved session {init_sid} for {key}")

    logger.info(f"Claude response: {reply[:30]}")


async def main():
    handler = AsyncSocketModeHandler(app, CREDS["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
