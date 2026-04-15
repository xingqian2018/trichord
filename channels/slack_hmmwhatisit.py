"""Slack channel integration using Socket Mode."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .base import Channel, InboundMessage, OutboundMessage


def _import_slack_sdk():
    """Lazy-import slack_sdk so the rest of the package works without it."""
    try:
        from slack_sdk.socket_mode.aiohttp import SocketModeClient
        from slack_sdk.socket_mode.request import SocketModeRequest
        from slack_sdk.socket_mode.response import SocketModeResponse
        from slack_sdk.web.async_client import AsyncWebClient

        return AsyncWebClient, SocketModeClient, SocketModeRequest, SocketModeResponse
    except ImportError:
        raise ImportError("Slack integration requires slack-sdk. Run: pip install slack-sdk")


_CANCEL_KEYWORDS = frozenset({"stop", "cancel", "abort"})


@runtime_checkable
class SessionManager(Protocol):
    """Minimal protocol for the object that handles message routing.

    Pass any object implementing these two methods — no koi dependency needed.
    """

    async def route_message(self, msg: InboundMessage) -> str | None: ...
    def cancel_session(self, session_key: str) -> bool: ...


class SlackChannel(Channel):
    """Slack channel using Socket Mode (no public URL needed).

    Behaviours:
    - Ack with eyes reaction while processing
    - Always reply in thread
    - Mention-gated in channels (@mention required), respond to all DMs
    - Show typing indicator while agent is working
    - Cancel running tasks by sending "stop", "cancel", or "abort" in-thread
    - Session key format:
        DM:      slack:dm:<user_id>:thread:<thread_ts>
        Channel: slack:channel:<channel_id>:thread:<thread_ts>
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        session_manager: Any,
        *,
        mention_only_in_channels: bool = True,
        ack_reaction: str = "eyes",
        allowed_users: str | list[str] | None = None,
    ):
        _import_slack_sdk()  # validate dependency is installed

        self._bot_token = bot_token
        self._app_token = app_token
        self._session_manager = session_manager
        self._mention_only = mention_only_in_channels
        self._ack_reaction = ack_reaction
        if allowed_users == "all":
            self._allowed_users = "all"
        elif isinstance(allowed_users, list):
            self._allowed_users = set(allowed_users)
        else:
            self._allowed_users = None  # default: deny all
        self._bot_user_id: str | None = None
        self._web = None
        self._socket = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        AsyncWebClient, SocketModeClient, _, _ = _import_slack_sdk()  # noqa: N806

        self._web = AsyncWebClient(token=self._bot_token)
        self._socket = SocketModeClient(app_token=self._app_token, web_client=self._web)

        resp = await self._web.auth_test()
        self._bot_user_id = resp["user_id"]
        logger.info("Slack bot user ID: {}", self._bot_user_id)

        # Register the event handler and connect
        self._socket.socket_mode_request_listeners.append(self._on_socket_event)
        await self._socket.connect()
        logger.info("Slack Socket Mode connected")

    async def stop(self) -> None:
        await self._socket.close()
        logger.info("Slack Socket Mode disconnected")

    # ------------------------------------------------------------------
    # Outbound
    # ------------------------------------------------------------------

    async def send_message(self, message: OutboundMessage) -> None:
        """Post a message to Slack (always in-thread)."""
        parts = message.session_key.split(":")
        # parts[2] is always the channel/user_id in both formats
        channel = parts[2]

        kwargs: dict[str, Any] = {"channel": channel, "text": message.text}
        if message.thread_id:
            kwargs["thread_ts"] = message.thread_id

        await self._web.chat_postMessage(**kwargs)

    # ------------------------------------------------------------------
    # Inbound event handling
    # ------------------------------------------------------------------

    async def _on_socket_event(self, client, req) -> None:
        """Handle a raw Socket Mode request."""
        _, _, SocketModeRequest, SocketModeResponse = _import_slack_sdk()  # noqa: N806

        # Always ack immediately to avoid Slack retries
        response = SocketModeResponse(envelope_id=req.envelope_id)
        await client.send_socket_mode_response(response)

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        await self._handle_event(event)

    async def _handle_event(self, event: dict) -> None:
        """Process a Slack event and route to the session manager."""
        event_type = event.get("type", "")

        # Ignore bot's own messages
        if event.get("bot_id") or event.get("user") == self._bot_user_id:
            return

        # Ignore message subtypes (edits, deletes, etc.)
        if event.get("subtype"):
            return

        # Only handle message and app_mention events
        if event_type not in ("message", "app_mention"):
            return

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        text = event.get("text", "").strip()
        thread_ts = event.get("thread_ts") or event.get("ts", "")
        ts = event.get("ts", "")
        channel_type = event.get("channel_type", "")

        if not text or not user_id:
            return

        if self._allowed_users is None:
            return
        if self._allowed_users != "all" and user_id not in self._allowed_users:
            return

        is_dm = channel_type == "im"
        is_mention = event_type == "app_mention" or (self._bot_user_id and f"<@{self._bot_user_id}>" in text)

        # Mention gating: in channels, only respond to @mentions or app_mention events
        if not is_dm and self._mention_only and not is_mention:
            return

        # Strip the bot mention from the text
        if self._bot_user_id:
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()

        if not text:
            return

        # Build session key
        if is_dm:
            session_key = f"slack:dm:{user_id}:thread:{thread_ts}"
        else:
            session_key = f"slack:channel:{channel_id}:thread:{thread_ts}"

        # Intercept cancel keywords before routing to the session manager
        if text.lower() in _CANCEL_KEYWORDS:
            cancelled = self._session_manager.cancel_session(session_key)
            if cancelled:
                asyncio.create_task(self._react_cancel(channel_id, ts))
            else:
                asyncio.create_task(self._send_cancel_reply(channel_id, ts, thread_ts, "Nothing running to cancel."))
            return

        inbound = InboundMessage(
            text=text,
            user_id=user_id,
            user_name=user_id,
            channel_id=channel_id,
            session_key=session_key,
            thread_id=thread_ts,
            source="slack",
            raw=event,
        )

        # Process in a task so we don't block the socket listener
        asyncio.create_task(self._process_message(inbound, channel_id, ts, thread_ts))

    async def _react_cancel(self, channel_id: str, ts: str) -> None:
        """Add an :x: reaction to acknowledge that cancellation was triggered."""
        try:
            await self._web.reactions_add(
                channel=channel_id,
                timestamp=ts,
                name="x",
            )
        except Exception:
            logger.opt(exception=True).debug("Could not add cancel reaction")

    async def _send_cancel_reply(
        self,
        channel_id: str,
        ts: str,
        thread_ts: str,
        text: str,
    ) -> None:
        """Post a reply when there is nothing running to cancel."""
        reply_thread = thread_ts or ts
        try:
            await self._web.chat_postMessage(
                channel=channel_id,
                text=text,
                thread_ts=reply_thread,
            )
        except Exception:
            logger.exception("Failed to send cancel reply in {}", channel_id)

    async def _process_message(
        self,
        msg: InboundMessage,
        channel_id: str,
        ts: str,
        thread_ts: str,
    ) -> None:
        """Ack, process, and reply to a message."""
        # Ack with reaction
        try:
            await self._web.reactions_add(
                channel=channel_id,
                timestamp=ts,
                name=self._ack_reaction,
            )
        except Exception:
            logger.opt(exception=True).debug("Could not add ack reaction")

        # Route message to session manager
        try:
            response_text = await self._session_manager.route_message(msg)
        except Exception:
            logger.exception("Error processing message in session {}", msg.session_key)
            response_text = "Sorry, I encountered an error processing your message."

        # Remove ack reaction
        try:
            await self._web.reactions_remove(
                channel=channel_id,
                timestamp=ts,
                name=self._ack_reaction,
            )
        except Exception:
            logger.opt(exception=True).debug("Could not remove ack reaction")

        # Reply in thread
        if response_text is not None:
            reply_thread = thread_ts or ts
            outbound = OutboundMessage(
                text=response_text,
                session_key=msg.session_key,
                thread_id=reply_thread,
            )
            try:
                await self.send_message(outbound)
            except Exception:
                logger.exception(
                    "Failed to send Slack reply for session {}",
                    msg.session_key,
                )
