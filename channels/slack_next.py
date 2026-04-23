import asyncio
import json
from pathlib import Path
from typing import Optional

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

SLACK_ROLLOVER_AT = 3000  # Seal current message and start a fresh one before hitting the limit
MAX_VISIBLE_TOOLS = 3  # How many in-flight tools to render before collapsing the rest

YES_ALWAYS = {"always", "always yes", "always_yes"}
YES = {"yes", "y", "allow", "approve", "ok", "sure"}
NO = {"no", "n", "deny", "reject", "stop"}
YES_ALL = {"yes all", "yesall", "allow all"}
NO_ALL = {"no all", "noall", "deny all"}

app = AsyncApp(token=CREDS["SLACK_BOT_TOKEN"])


ACTIVE_CLAUDE_SESSION: "dict[str, SlackClaudeSession]" = {}


def find_rollover_cut(text: str, soft_limit: int) -> int:
    if len(text) <= soft_limit:
        return len(text)
    head = text[:soft_limit]
    for delim in ('```', '"""'):
        if head.count(delim) % 2 == 1:
            opening = head.rfind(delim)
            if opening > 0:
                return opening
    for boundary in ('\n\n', '\n'):
        idx = head.rfind(boundary)
        if idx > 0:
            return idx + len(boundary)
    min_cut = soft_limit // 2
    for boundary in ('. ', ' '):
        idx = head.rfind(boundary)
        if idx >= min_cut:
            return idx + len(boundary)
    return soft_limit


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

        self.pending: dict[str, asyncio.Future[tuple[str, str]]] = {}
        self.pending_tool_verdicts: dict[str, tuple[str, str]] = {}
        self.pending_tool_tracker: dict[str, ToolUseBlock] = {}
        self.always_allow_pending_tool_flag = False

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
                        return ""
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

    def create_pending_tool_message(self) -> str:
        tools = list(self.pending_tool_tracker.values())
        if not tools:
            return ""
        hidden = max(0, len(tools) - MAX_VISIBLE_TOOLS)
        visible = tools[:MAX_VISIBLE_TOOLS] if hidden else tools
        lines = [f"\n> :wrench: `{t.name}` {json.dumps(t.input)[:120]}" for t in visible]
        if hidden:
            lines += [f"\n> _… and {hidden} more_"]
        return "\n".join(lines) + "\n"

    def create_permission_asking_message(self) -> str:
        return f":lock: Asking permission \n" + self.PERMISSION_QUESTION

    def create_permission_verdict_message(self, verdict: str, message: Optional[str] = None) -> str:
        message = message or "N/A"
        return {
            "always_allow": ":white_check_mark::white_check_mark::white_check_mark: Always Allowed",
            "allow_all": ":white_check_mark::white_check_mark: Allowed All",
            "allow": ":white_check_mark: Allowed",
            "deny_all": ":x::x: Denied All",
            "deny": ":x: Denied",
            "interrupt": f":arrow_right_hook: Denied because {message}",
        }.get(verdict, ":grey_question: Error")

    async def update_main(self):
        async with self.lock:
            if self.main_ts is None:
                await self.post_fresh_thinking()

            while len(self.response_str) > SLACK_ROLLOVER_AT:
                cut = find_rollover_cut(self.response_str, SLACK_ROLLOVER_AT)
                head = self.response_str[:cut]
                tail = self.response_str[cut:]
                try:
                    await app.client.chat_update(
                        channel=self.channel, ts=self.main_ts, text=head + "\n_(...continued below...)_",
                    )
                except Exception:
                    self.logger.exception("failed to seal main message for rollover")
                self.response_str = tail
                self.main_ts = None
                await self.post_fresh_thinking()

            body = self.response_str
            if self.create_pending_tool_message() and not (self.always_allow_pending_tool_flag):
                body += self.create_pending_tool_message()
                body += self.create_permission_asking_message()
            if not body:
                return
            try:
                await app.client.chat_update(
                    channel=self.channel, ts=self.main_ts, text=body,
                )
            except Exception:
                self.logger.exception("failed to seal last main message")
        return

    async def update_main_with_verdict(self, verdict: str, message: Optional[str] = None):
        async with self.lock:
            body = self.response_str
            body += self.create_pending_tool_message()
            body += self.create_permission_verdict_message(verdict, message)
            try:
                await app.client.chat_update(
                    channel=self.channel, ts=self.main_ts, text=body,
                )
                self.response_str = ""
                self.main_ts = None
                await self.post_fresh_thinking()
            except Exception:
                self.logger.exception("failed to seal last main message")

    async def post_fresh_thinking(self):
        assert self.main_ts is None
        if self.lock:
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

            if stripped in YES_ALWAYS:
                for tid in self.pending_tool_tracker:
                    self.always_allow_pending_tool_flag = True
                    self.pending_tool_verdicts[tid] = ("always_allow", "user always allowed")
            elif stripped in YES_ALL:
                for tid in self.pending_tool_tracker:
                    self.pending_tool_verdicts[tid] = ("allow_all", "user allowed all pending")
            elif stripped in YES:
                first_id = next(iter(self.pending))
                self.pending_tool_verdicts[first_id] = ("allow", "user allowed current pending")
            elif stripped in NO_ALL:
                for tid in self.pending_tool_tracker:
                    self.pending_tool_verdicts[tid] = ("deny_all", "user denied all pending")
            elif stripped in NO:
                first_id = next(iter(self.pending))
                self.pending_tool_verdicts[first_id] = ("deny", "user denied current pending")
            else:
                for i, tid in enumerate(list(self.pending)):
                    if i == 0:
                        self.pending_tool_verdicts[tid] = ("interrupt", text)
                    else:
                        self.pending_tool_verdicts[tid] = ("deny", "user interrupted")

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
            if self.always_allow_pending_tool_flag:
                synthesized = ("allow", "user allowed")

        if synthesized is None:
            fut: asyncio.Future[tuple[str, str]] = loop.create_future()
            async with self.lock:
                self.pending[tool_use_id] = fut
            await self.update_main()

            try:
                verdict, message = await fut
            except asyncio.CancelledError:
                async with self.lock:
                    self.pending.pop(tool_use_id, None)
                await self.update_main_with_verdict("error")
                return PermissionResultDeny(message="permission cancelled")

            await self.update_main_with_verdict(verdict, message)
        else:
            verdict, message = synthesized

        if verdict == "allow":
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
                    await self.update_main()
                    async with self.lock:
                        self.response_str = ""
                        self.main_ts = None
                    break
                fragment = self.render_message(msg)
                if fragment:
                    async with self.lock:
                        self.response_str += "\n" + fragment
                await self.update_main()


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
