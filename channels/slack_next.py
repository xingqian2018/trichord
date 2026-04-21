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

# thread_key -> FIFO of asyncio.Future[tuple[str, str]]
# ("allow", "")            -> approve
# ("deny", "...reason")    -> deny with reason surfaced to claude
# ("interrupt", "...text") -> deny + interrupt claude so text becomes next-turn input
PENDING: dict[str, deque[asyncio.Future]] = {}

# thread_key -> True if "yes all" is in effect for the rest of this Claude turn
# (cleared automatically in drive_claude's finally)
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


def extract_init(msg) -> tuple[str | None, str | None]:
    if isinstance(msg, SystemMessage):
        data = getattr(msg, "data", {}) or {}
        return data.get("session_id"), data.get("model")
    return None, None


async def drive_claude(
    text: str,
    session_id: str | None,
    channel: str,
    thread_ts: str,
    initial_ts: str,
    gear_ts: str | None,
    logger,
) -> tuple[str, str | None]:
    """Run claude and surface events as discrete Slack messages.

    `initial_ts` is the first hourglass placeholder (posted by the caller).
    For each event with renderable content, we overwrite the current hourglass
    with that content and post a fresh hourglass below it. When the turn ends,
    we delete the trailing hourglass.
    """
    accumulated = ""  # kept only for the logger at the end of the turn
    init_sid: str | None = None
    # active_ts is the hourglass Slack ts. Invariant: set iff we are waiting
    # on Claude. Cleared when Claude is blocked on a user permission reply.
    active_ts: str | None = initial_ts
    # Buffer for consecutive render fragments. Flushed at natural boundaries
    # (before a permission prompt, at turn end) so a run of tool uses / results
    # collapses into a single Slack message instead of one per event.
    buffer = ""
    # If the user replies to a permission prompt with arbitrary text, we deny
    # the tool (with interrupt=True) and stash the text here so the outer loop
    # can send it as the next turn's user message.
    pending_followup: str | None = None
    key = thread_key(channel, thread_ts)
    loop = asyncio.get_running_loop()

    async def post_hourglass() -> None:
        nonlocal active_ts
        if active_ts is not None:
            return
        try:
            msg = await app.client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=":hourglass_flowing_sand: _thinking..._",
            )
            active_ts = msg["ts"]
        except Exception as e:
            logger.warning(f"post hourglass failed: {e}")

    async def remove_hourglass() -> None:
        nonlocal active_ts
        if active_ts is None:
            return
        try:
            await app.client.chat_delete(channel=channel, ts=active_ts)
        except Exception as e:
            logger.warning(f"delete hourglass failed: {e}")
        active_ts = None

    async def commit(body: str) -> None:
        """Turn the current hourglass into `body`, then open a fresh one below."""
        nonlocal active_ts
        if len(body) > SLACK_MAX_TEXT:
            body = "_(...earlier output truncated...)_\n\n" + body[-SLACK_MAX_TEXT:]
        if active_ts is not None:
            try:
                await app.client.chat_update(channel=channel, ts=active_ts, text=body)
            except Exception as e:
                logger.warning(f"chat_update failed: {e}")
            active_ts = None
        else:
            try:
                await app.client.chat_postMessage(
                    channel=channel, thread_ts=thread_ts, text=body
                )
            except Exception as e:
                logger.warning(f"chat_postMessage failed: {e}")
        await post_hourglass()

    async def flush_buffer() -> None:
        """Commit buffered fragments as a single Slack message."""
        nonlocal buffer
        if not buffer:
            return
        body, buffer = buffer, ""
        await commit(body)

    async def can_use_tool(tool_name, tool_input, _context):
        if AUTO_APPROVE.get(key):
            return PermissionResultAllow(updated_input=tool_input)

        # Flush any pending content so the :lock: prompt lands at the bottom.
        await flush_buffer()
        # Claude is now blocked on us, not the other way around -> hide hourglass.
        await remove_hourglass()

        fut: asyncio.Future[tuple[str, str]] = loop.create_future()
        PENDING.setdefault(key, deque()).append(fut)
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
            await post_hourglass()
            return PermissionResultDeny(message="permission cancelled")

        if prompt_ts is not None:
            try:
                await app.client.chat_delete(channel=channel, ts=prompt_ts)
            except Exception as e:
                logger.warning(f"delete permission prompt failed: {e}")

        # User replied -> claude is about to work again, restore hourglass.
        await post_hourglass()

        if verdict == "allow":
            return PermissionResultAllow(updated_input=tool_input)
        if verdict == "interrupt":
            # Stash the user's text so the outer loop can resend it as a new
            # user turn after claude's current turn unwinds from the interrupt.
            nonlocal pending_followup
            pending_followup = message
            return PermissionResultDeny(message=message, interrupt=True)
        return PermissionResultDeny(message=message or "user denied")

    options = ClaudeAgentOptions(resume=session_id, can_use_tool=can_use_tool)
    try:
        async with ClaudeSDKClient(options=options) as client:
            next_text: str | None = text
            while next_text is not None:
                current = next_text
                next_text = None
                await client.query(current)
                last_role: str | None = None
                async for msg in client.receive_response():
                    sid, model = extract_init(msg)
                    if sid and init_sid is None:
                        init_sid = sid
                        if gear_ts is not None:
                            try:
                                await app.client.chat_update(
                                    channel=channel,
                                    ts=gear_ts,
                                    text=f":gear: `{model or '?'}` / `{sid}`",
                                )
                            except Exception as e:
                                logger.warning(f"gear chat_update failed: {e}")
                        continue
                    if isinstance(msg, ResultMessage):
                        continue
                    fragment = render_message(msg)
                    if not fragment:
                        continue
                    # Flush the previous batch when role changes
                    # (assistant text/tool_use <-> user tool_result).
                    role = "assistant" if isinstance(msg, AssistantMessage) else "user"
                    if last_role is not None and role != last_role:
                        await flush_buffer()
                    last_role = role
                    accumulated += fragment
                    buffer += fragment
                # Turn ended for this query; flush whatever is left.
                await flush_buffer()
                # If the user interrupted with a follow-up, feed it back in.
                if pending_followup is not None:
                    next_text = pending_followup
                    pending_followup = None
    except Exception as e:
        logger.exception("claude query failed")
        buffer += f"\n:x: `{e}`"
        await flush_buffer()
        raise
    finally:
        AUTO_APPROVE.pop(key, None)

    # Turn ended. Clean up any dangling hourglass.
    if active_ts is not None:
        if accumulated:
            try:
                await app.client.chat_delete(channel=channel, ts=active_ts)
            except Exception as e:
                logger.warning(f"delete hourglass failed: {e}")
        else:
            try:
                await app.client.chat_update(
                    channel=channel, ts=active_ts, text="_(no response)_"
                )
            except Exception as e:
                logger.warning(f"chat_update failed: {e}")

    return accumulated, init_sid


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
            # Approve the currently-pending prompt AND every future tool call
            # for the rest of this Claude turn (cleared in drive_claude's finally).
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

        # Any other text: interrupt claude with the user's message as follow-up input.
        while queue:
            fut = queue.popleft()
            if not fut.done():
                fut.set_result(("interrupt", text))
        return

    logger.info(f"DM from user={user} channel={channel}: {text[:30]}")

    log = load_log()
    session_id = log.get(key)
    is_new_session = not session_id

    gear_msg = None
    if is_new_session:
        gear_msg = await say(text=":gear: _starting new session..._", thread_ts=thread_ts)

    placeholder = await say(text=":hourglass_flowing_sand: _thinking..._", thread_ts=thread_ts)
    initial_ts = placeholder["ts"]

    if session_id:
        logger.info(f"Resuming claude session {session_id} for {key}")
    else:
        logger.info(f"Starting new claude session for {key}")

    gear_ts = gear_msg["ts"] if gear_msg else None
    reply, init_sid = await drive_claude(
        text, session_id, channel, thread_ts, initial_ts, gear_ts, logger
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
