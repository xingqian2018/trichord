#!/usr/bin/env python3
"""
cslack helper — read messages and post replies to a Slack channel.

Usage:
  python slack_client.py get   --channel CHANNEL [--limit N] [--thread THREAD_TS]
  python slack_client.py reply --channel CHANNEL --text TEXT [--thread THREAD_TS]
  python slack_client.py info  --channel CHANNEL

Credentials: <project_root>/credentials/slack.json
  {
    "SLACK_BOT_TOKEN":  "xoxb-...",   # required for sending / reading
    "SLACK_USER_TOKEN": "xoxp-...",   # optional; used for reading if bot lacks scopes
    "SLACK_APP_TOKEN":  "xapp-...",   # unused here; read by channels/slack.py bot
    "default_channel":  "#my-channel" # optional
  }

Legacy lowercase keys (bot_token / user_token) are also accepted for backward compat.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


def resolve_credentials_path() -> Path:
    """Locate credentials/slack.json. Prefer a path relative to this script
    (works when script lives in <root>/cc/cslack/), then fall back to walking
    up from CWD.
    """
    script = Path(__file__).resolve()
    by_script = script.parent.parent.parent / "credentials" / "slack.json"
    if by_script.exists():
        return by_script
    for p in [Path.cwd(), *Path.cwd().parents]:
        cand = p / "credentials" / "slack.json"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"credentials/slack.json not found (tried {by_script} and upward from {Path.cwd()})"
    )


def load_config() -> dict:
    try:
        path = resolve_credentials_path()
    except FileNotFoundError as e:
        print(json.dumps({"error": str(e), "success": False}))
        sys.exit(1)
    raw = json.loads(path.read_text())
    return {
        "bot_token":       raw.get("SLACK_BOT_TOKEN")  or raw.get("bot_token"),
        "user_token":      raw.get("SLACK_USER_TOKEN") or raw.get("user_token"),
        "default_channel": raw.get("default_channel"),
    }


def require_sdk():
    try:
        from slack_sdk.web.async_client import AsyncWebClient
        return AsyncWebClient
    except ImportError:
        print(json.dumps({"error": "slack-sdk not installed. Run: pip install slack-sdk", "success": False}))
        sys.exit(1)


async def resolve_channel(client, name: str) -> str | None:
    """Resolve channel name (#foo or foo) to channel ID."""
    name = name.lstrip("#")
    # If already looks like a Slack ID, pass through
    if name.startswith(("C", "G", "D", "U")) and len(name) >= 9:
        return name
    cursor = None
    while True:
        # Only request types our bot token has scope for. `im` would require
        # `im:read`, which most bots don't have — listing DMs by name is rarely
        # useful anyway (DMs don't have names; pass a raw D... ID instead,
        # which is handled by the short-circuit above).
        kwargs: dict[str, Any] = {"limit": 200, "types": "public_channel,private_channel,mpim"}
        if cursor:
            kwargs["cursor"] = cursor
        resp = await client.conversations_list(**kwargs)
        for ch in resp.get("channels", []):
            if ch.get("name") == name:
                return ch["id"]
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return None


async def resolve_user(client, who: str) -> str | None:
    """Resolve a user reference to a Slack user ID.

    Accepts:
      - Raw user ID like `U01ABCDEFGH` (length ≥ 9, starts with U/W) → passthrough
      - `@username` or `username` → looked up via users.list against `name` and `real_name`
      - Email like `a@b.com` → tried via users.lookupByEmail (needs users:read.email),
        falls back to scanning users.list for a profile.email match
    Returns None if not found.
    """
    who = who.lstrip("@").strip()
    if not who:
        return None
    # Raw ID passthrough
    if who[0] in ("U", "W") and len(who) >= 9 and who.replace("_", "").isalnum():
        return who
    is_email = "@" in who
    if is_email:
        try:
            resp = await client.users_lookupByEmail(email=who)
            u = resp.get("user") or {}
            if u.get("id"):
                return u["id"]
        except Exception:
            pass  # fall through to list scan — may still find via profile.email
    cursor = None
    lower = who.lower()
    while True:
        kwargs: dict[str, Any] = {"limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = await client.users_list(**kwargs)
        for u in resp.get("members", []):
            if u.get("deleted"):
                continue
            if is_email:
                if (u.get("profile") or {}).get("email", "").lower() == lower:
                    return u.get("id")
            else:
                if u.get("name", "").lower() == lower or u.get("real_name", "").lower() == lower:
                    return u.get("id")
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return None


async def get_messages(config: dict, channel: str, limit: int, thread_ts: str | None) -> dict:
    AsyncWebClient = require_sdk()

    # Prefer user_token for reading (bot tokens may not have history access in all setups)
    token = config.get("user_token") or config.get("bot_token")
    if not token:
        return {"error": "No token in config (need user_token or bot_token)", "success": False}

    client = AsyncWebClient(token=token)
    channel_id = await resolve_channel(client, channel)
    if not channel_id:
        return {"error": f"Channel not found: {channel}", "success": False}

    limit = min(limit, 200)
    kwargs: dict[str, Any] = {"channel": channel_id, "limit": limit}

    try:
        if thread_ts:
            kwargs["ts"] = thread_ts
            resp = await client.conversations_replies(**kwargs)
        else:
            resp = await client.conversations_history(**kwargs)

        messages = resp.get("messages", [])
        formatted = []
        for msg in messages:
            user = msg.get("user", msg.get("bot_id", "unknown"))
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            reply_count = msg.get("reply_count", 0)
            entry: dict[str, Any] = {"ts": ts, "user": user, "text": text}
            if reply_count:
                entry["reply_count"] = reply_count
            # Subtype (e.g. bot_message)
            if msg.get("subtype"):
                entry["subtype"] = msg["subtype"]
            if msg.get("bot_id"):
                entry["bot_id"] = msg["bot_id"]
            formatted.append(entry)

        return {
            "messages": formatted,
            "count": len(formatted),
            "has_more": resp.get("has_more", False),
            "channel_id": channel_id,
            "success": True,
        }
    except Exception as e:
        return {"error": f"Slack API error: {e}", "success": False}


async def reply_message(
    config: dict,
    channel: str,
    text: str,
    thread_ts: str | None,
    mention: list[str] | None = None,
) -> dict:
    AsyncWebClient = require_sdk()

    token = config.get("bot_token")
    if not token:
        return {"error": "bot_token required in config for sending messages", "success": False}

    client = AsyncWebClient(token=token)
    channel_id = await resolve_channel(client, channel)
    if not channel_id:
        return {"error": f"Channel not found: {channel}", "success": False}

    # Resolve @mentions. Each entry may be a user ID, @name, name, or email.
    mention_ids: list[str] = []
    mention_failed: list[str] = []
    if mention:
        for m in mention:
            m = m.strip()
            if not m:
                continue
            uid = await resolve_user(client, m)
            if uid:
                mention_ids.append(uid)
            else:
                mention_failed.append(m)
    if mention_ids:
        # Slack renders <@UID> as a clickable @mention. Prepend so the text still
        # reads naturally even if the message body also references the user.
        prefix = " ".join(f"<@{uid}>" for uid in mention_ids)
        text = f"{prefix} {text}"

    kwargs: dict[str, Any] = {"channel": channel_id, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts

    try:
        resp = await client.chat_postMessage(**kwargs)
        result = {
            "channel_id": channel_id,
            "ts": resp.get("ts"),
            "thread_ts": thread_ts,
            "success": True,
        }
        if mention_ids:
            result["mentions"] = mention_ids
        if mention_failed:
            result["mention_unresolved"] = mention_failed
        return result
    except Exception as e:
        return {"error": f"Slack API error: {e}", "success": False}


async def channel_info(config: dict, channel: str) -> dict:
    AsyncWebClient = require_sdk()

    token = config.get("user_token") or config.get("bot_token")
    if not token:
        return {"error": "No token in config", "success": False}

    client = AsyncWebClient(token=token)
    channel_id = await resolve_channel(client, channel)
    if not channel_id:
        return {"error": f"Channel not found: {channel}", "success": False}

    try:
        resp = await client.conversations_info(channel=channel_id)
        ch = resp.get("channel", {})
        return {
            "id": ch.get("id"),
            "name": ch.get("name"),
            "is_private": ch.get("is_private"),
            "is_im": ch.get("is_im"),
            "member_count": ch.get("num_members"),
            "topic": ch.get("topic", {}).get("value"),
            "purpose": ch.get("purpose", {}).get("value"),
            "success": True,
        }
    except Exception as e:
        return {"error": f"Slack API error: {e}", "success": False}


def main():
    parser = argparse.ArgumentParser(description="cslack — Slack get/reply helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # get subcommand
    p_get = subparsers.add_parser("get", help="Fetch messages from a channel")
    p_get.add_argument("--channel", required=True, help="Channel name (e.g. #general) or ID")
    p_get.add_argument("--limit", type=int, default=20, help="Max messages to fetch (default: 20)")
    p_get.add_argument("--thread", default=None, help="Thread timestamp to fetch replies for")

    # reply subcommand
    p_reply = subparsers.add_parser("reply", help="Post a message or reply to a thread")
    p_reply.add_argument("--channel", required=True, help="Channel name or ID")
    p_reply.add_argument("--text", required=True, help="Message text to send")
    p_reply.add_argument("--thread", default=None, help="Thread timestamp to reply into")
    p_reply.add_argument(
        "--mention",
        action="append",
        default=None,
        help=(
            "User to @-mention (prepended to text). Accepts a raw user ID "
            "(U01ABCDEFGH), @username, username, or email. May be passed multiple "
            "times to mention several users."
        ),
    )

    # info subcommand
    p_info = subparsers.add_parser("info", help="Get channel metadata")
    p_info.add_argument("--channel", required=True, help="Channel name or ID")

    args = parser.parse_args()
    config = load_config()

    if args.command == "get":
        result = asyncio.run(get_messages(config, args.channel, args.limit, args.thread))
    elif args.command == "reply":
        result = asyncio.run(
            reply_message(config, args.channel, args.text, args.thread, args.mention)
        )
    elif args.command == "info":
        result = asyncio.run(channel_info(config, args.channel))
    else:
        result = {"error": f"Unknown command: {args.command}", "success": False}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
