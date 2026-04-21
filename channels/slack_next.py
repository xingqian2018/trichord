import asyncio
import json
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

SLACK_MAX_TEXT = 38000  # Slack chat.update hard limit is 40000 chars

YES = {"yes", "y", "allow", "approve", "ok", "sure"}
NO = {"no", "n", "deny", "reject", "stop"}
YES_ALL = {"yes all", "yesall", "allow all"}
NO_ALL = {"no all", "noall", "deny all"}

app = AsyncApp(token=CREDS["SLACK_BOT_TOKEN"])


ACTIVE_CLAUDE_SESSION: "dict[str, SlackClaudeSession]" = {}


class SlackClaudeSession:
    PERMISSION_QUESTION = (
        "_reply *yes* / *no* / *yes all* / *no all* "
        "— any other text interrupts claude with your message_"
    )

    def __init__(self, channel: str, thread_ts: str, logger):
        self.channel = channel
        self.thread_ts = thread_ts
        self.key = f"{channel}:{thread_ts}"
        self.logger = logger
        self.lock = asyncio.Lock()

        self.session_id: str | None = None
        self.header_ts: str | None = None
        self.main_ts: str | None = None
        self.response_str: str = ""
        # tool_use_id -> Future[tuple[verdict, message]] — tools whose
        # can_use_tool has been called and is awaiting a decision.
        self.pending: dict[str, asyncio.Future[tuple[str, str]]] = {}
        # tool_use_id -> (verdict, message) — pre-computed decisions for tools
        # that haven't reached can_use_tool yet (e.g. seeded by "yes all").
        self.pending_tool_verdicts: dict[str, tuple[str, str]] = {}
        # tool_use_id -> ToolUseBlock — tools Claude has called but whose
        # result has not arrived yet.
        self.pending_tool_tracker: dict[str, ToolUseBlock] = {}

    def has_pending(self) -> bool:
        return bool(self.pending)

    def render_message(self, msg) -> str:
        if isinstance(msg, AssistantMessage):
            parts = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    parts.append(block.text)
                elif isinstance(block, ToolUseBlock):
                    self.pending_tool_tracker[block.id] = block
                    line = f":wrench: `{block.name}` {json.dumps(block.input)[:120]}"
                    parts.append(f"\n> {line}\n")
                elif isinstance(block, ThinkingBlock):
                    pass
            return "".join(parts)
        if isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        self.pending_tool_tracker.pop(block.tool_use_id, None)
                        self.pending_tool_verdicts.pop(block.tool_use_id, None)
                        result = block.content
                        if isinstance(result, list):
                            result = " ".join(c.get("text", "") for c in result if isinstance(c, dict))
                        snippet = str(result or "").strip().replace("\n", " ")[:160]
                        return f"> _result:_ {snippet}\n"
        return ""

    async def try_finalize_gear(self, msg) -> None:
        if self.header_ts is None or not isinstance(msg, SystemMessage):
            return
        sid = msg.data.get('session_id')
        model = msg.data.get('model')
        if not sid:
            return
        try:
            await app.client.chat_update(
                channel=self.channel, ts=self.header_ts,
                text=f":gear: `{model or '?'}` / `{sid}`",
            )
        except Exception:
            self.logger.exception("failed to finalize gear header")
        self.header_ts = None

    def create_permission_asking_message(self) -> str:
        return f":lock: Asking permission ({len(self.pending_tool_tracker)} in flight)\n" + self.PERMISSION_QUESTION

    def create_permission_verdict_message(self, verdict: str) -> str:
        return {
            "allow_all": ":white_check_mark: Allowed All",
            "allow": ":white_check_mark: Allowed",
            "deny_all": ":x: Denied All",
            "deny": ":x: Denied",
            "interrupt": ":arrow_right_hook: Denied with reasoning",
        }.get(verdict, ":grey_question: Error")

    async def _update_main(self, body: str):
        # Caller must hold self.lock.
        if len(body) > SLACK_MAX_TEXT:
            body = "_(...earlier output truncated...)_\n\n" + body[-SLACK_MAX_TEXT:]
        try:
            if self.main_ts is None:
                posted = await app.client.chat_postMessage(
                    channel=self.channel, thread_ts=self.thread_ts, text=body,
                )
                self.main_ts = posted["ts"]
            else:
                await app.client.chat_update(
                    channel=self.channel, ts=self.main_ts, text=body,
                )
        except Exception:
            self.logger.exception("failed to update main message")

    async def refresh_asking_permission(self):
        await self._update_main(self.response_str + self.create_permission_asking_message())

    async def refresh_with_permission(self, verdict: str):
        await self._update_main(self.response_str + self.create_permission_verdict_message(verdict))

    async def post_fresh_thinking(self):
        try:
            msg = await app.client.chat_postMessage(
                channel=self.channel, thread_ts=self.thread_ts,
                text=":hourglass_flowing_sand: _thinking..._",
            )
            self.main_ts = msg["ts"]
        except Exception:
            self.logger.exception("failed to post fresh thinking placeholder")

    async def deliver_response(self, text: str) -> bool:
        stripped = text.strip().lower()
        async with self.lock:
            if not self.pending:
                return False

            if stripped in YES_ALL:
                # Assign allow_all to every known in-flight tool_use_id.
                for tid in self.pending_tool_tracker:
                    self.pending_tool_verdicts[tid] = ("allow_all", "user allowed")
            elif stripped in NO_ALL:
                for tid in self.pending_tool_tracker:
                    self.pending_tool_verdicts[tid] = ("deny_all", "user denied")
            elif stripped in YES:
                # First waiting fut gets allow.
                first_id = next(iter(self.pending))
                self.pending_tool_verdicts[first_id] = ("allow", "user allowed")
            elif stripped in NO:
                first_id = next(iter(self.pending))
                self.pending_tool_verdicts[first_id] = ("deny", "user denied")
            else:
                # Interrupt: first waiting fut carries the user's text and
                # aborts the turn; the rest deny quietly.
                for i, tid in enumerate(list(self.pending)):
                    if i == 0:
                        self.pending_tool_verdicts[tid] = ("interrupt", text)
                    else:
                        self.pending_tool_verdicts[tid] = ("deny", "user interrupted")

            # Resolve any waiting fut whose verdict is now decided.
            for tid in list(self.pending.keys()):
                if tid in self.pending_tool_verdicts:
                    fut = self.pending.pop(tid)
                    verdict = self.pending_tool_verdicts.pop(tid)
                    if not fut.done():
                        fut.set_result(verdict)
        return True

    async def can_use_tool(self, tool_name, tool_input, context):
        loop = asyncio.get_running_loop()
        tool_use_id = context.tool_use_id

        synthesized: tuple[str, str] | None = None
        async with self.lock:
            if tool_use_id in self.pending_tool_verdicts:
                synthesized = self.pending_tool_verdicts.pop(tool_use_id)

        if synthesized is None:
            fut: asyncio.Future[tuple[str, str]] = loop.create_future()
            async with self.lock:
                self.pending[tool_use_id] = fut
                await self.refresh_asking_permission()

            try:
                verdict, message = await fut
            except asyncio.CancelledError:
                async with self.lock:
                    self.pending.pop(tool_use_id, None)
                    await self.refresh_with_permission("error")
                return PermissionResultDeny(message="permission cancelled")

            async with self.lock:
                await self.refresh_with_permission(verdict)
                self.response_str = ""
                await self.post_fresh_thinking()
        else:
            verdict, message = synthesized

        if verdict in ("allow", "allow_all"):
            return PermissionResultAllow(updated_input=tool_input)
        if verdict == "interrupt":
            return PermissionResultDeny(message=message, interrupt=True)
        return PermissionResultDeny(message=message or "user denied")

    async def run(self, text: str) -> None:
        async with self.lock:
            self.main_ts = None
            self.response_str = ""
            self.pending.clear()
            self.pending_tool_verdicts.clear()
            self.pending_tool_tracker.clear()

            if self.session_id is None and self.header_ts is None:
                try:
                    header_msg = await app.client.chat_postMessage(
                        channel=self.channel, thread_ts=self.thread_ts,
                        text=":gear: _starting new session..._",
                    )
                    self.header_ts = header_msg['ts']
                except Exception:
                    self.logger.exception("failed to post gear header")

            try:
                msg = await app.client.chat_postMessage(
                    channel=self.channel, thread_ts=self.thread_ts,
                    text=":hourglass_flowing_sand: _thinking..._",
                )
                self.main_ts = msg["ts"]
            except Exception:
                self.logger.exception("failed to post initial placeholder")

        options = ClaudeAgentOptions(
            resume=self.session_id,
            can_use_tool=self.can_use_tool,
        )

        async with ClaudeSDKClient(options=options) as client:
            await client.query(text)
            async for msg in client.receive_response():
                if isinstance(msg, SystemMessage):
                    sid = msg.data.get('session_id')
                    if sid and self.session_id is None:
                        self.session_id = sid
                await self.try_finalize_gear(msg)
                if isinstance(msg, ResultMessage):
                    async with self.lock:
                        if self.response_str:
                            await self._update_main(self.response_str)
                    break
                fragment = self.render_message(msg)
                if fragment:
                    async with self.lock:
                        self.response_str += fragment
                        if self.pending:
                            await self.refresh_asking_permission()
                        else:
                            await self._update_main(self.response_str)


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
    key = f"{channel}:{thread_ts}"

    session = ACTIVE_CLAUDE_SESSION.get(key)

    if session is not None and session.has_pending():
        await session.deliver_response(text)
        return

    logger.info(f"DM from user={user} channel={channel}: {text[:30]}")

    if session is None:
        session = SlackClaudeSession(channel, thread_ts, logger)
        ACTIVE_CLAUDE_SESSION[key] = session

    await session.run(text)


async def main():
    handler = AsyncSocketModeHandler(app, CREDS["SLACK_APP_TOKEN"])
    await handler.start_async()


if __name__ == "__main__":
    asyncio.run(main())
