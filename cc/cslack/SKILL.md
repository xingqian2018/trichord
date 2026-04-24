---
name: cslack
description: Get messages and reply to a specific Slack channel or thread using the Slack API.
user_invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

## Purpose

Read messages from and post replies to a Slack channel or thread. Operates against a configured Slack app (bot token + optional user token) — independent of KoI.

## Config

Credentials live in the project's unified credentials file at `<project_root>/credentials/slack.json` — the same file the `channels/slack.py` bot reads:

```json
{
  "SLACK_BOT_TOKEN":  "xoxb-...",
  "SLACK_USER_TOKEN": "xoxp-...",
  "SLACK_APP_TOKEN":  "xapp-...",
  "default_channel":  "#your-channel"
}
```

- **`SLACK_BOT_TOKEN`** — required for sending (posting messages). Create a Slack app, add `chat:write` scope, install to workspace, copy Bot OAuth token. Also used as a fallback for reading if the bot has `channels:history` / `groups:history` / `mpim:history` / `im:history` scopes.
- **`SLACK_USER_TOKEN`** — optional; only needed if your bot can't read history in the channels you care about. A user token (`xoxp-…`) reads as the installing user and bypasses per-channel bot membership.
- **`SLACK_APP_TOKEN`** — not used by this skill; lives in the same file for the `channels/slack.py` Socket Mode bot.
- **`default_channel`** — optional fallback when no channel is specified.

Legacy lowercase keys (`bot_token`, `user_token`) are still accepted for backward compatibility — the loader normalizes them.

The helper resolves `credentials/slack.json` by walking up from the script's own directory first, then from CWD — so it works regardless of where you invoke it from. If the file is missing, the client prints a clear error showing the paths it tried.

## Helper Script

`slack_client.py` in this folder is a standalone CLI that wraps the Slack SDK.
It is **self-contained** — no dependency on imaginaire4 or KoI.

Install dependency once if needed:
```bash
pip install slack-sdk
```

### Subcommands

```
# Fetch the last N messages from a channel
python cc/cslack/slack_client.py get --channel "#general" --limit 20

# Fetch replies in a thread
python cc/cslack/slack_client.py get --channel "#general" --thread 1712345678.123456

# Post a message to a channel
python cc/cslack/slack_client.py reply --channel "#general" --text "Hello!"

# Post a reply into a thread
python cc/cslack/slack_client.py reply --channel "#general" --thread 1712345678.123456 --text "Got it, thanks."

# Post a message that @-mentions someone (repeat --mention for multiple users).
# Accepts a raw user ID (U01ABCDEFGH), @name, name, or email.
python cc/cslack/slack_client.py reply --channel "#general" \
    --mention U01ABCDEFGH --mention alice \
    --text "heads up — deploy finished"

# Get channel metadata (ID, member count, topic)
python cc/cslack/slack_client.py info --channel "#general"
```

All output is JSON. `"success": true` on success, `"error": "..."` on failure.

## Skill Invocation

When invoked as `/cslack`, interpret the user's natural language instruction:

### Examples

| User says | Action |
|-----------|--------|
| `/cslack get #dev` | Fetch last 20 messages from #dev |
| `/cslack get #dev last 50` | Fetch last 50 messages |
| `/cslack reply #dev "Looks good!"` | Post message to #dev |
| `/cslack reply #dev @alice "heads up — deploy finished"` | Post to #dev and @-mention alice (adds `--mention alice`) |
| `/cslack reply to thread 1712345678.123 in #dev "Done"` | Reply into that thread |
| `/cslack get thread 1712345678.123 in #dev` | Fetch all replies in that thread |
| `/cslack info #dev` | Show channel metadata |

### Default channel fallback

If the user does not specify a channel, read `default_channel` from config. If that is also missing, ask the user.

## Execution Steps

1. **Parse intent** — extract command (`get` / `reply` / `info`), channel, text (if reply), thread_ts (if thread), and limit.
2. **Run the helper** — call `python cc/cslack/slack_client.py <subcommand> [args]` from the project root.
3. **Parse JSON output** — check `"success"` field. If false, surface the `"error"` message clearly.
4. **Present results**:
   - For `get`: display messages in a readable format, one per line: `[ts] user: text (N replies)`
   - For `reply`: confirm `"Message sent — ts: <ts>"`.
   - For `info`: display channel name, ID, member count, topic.

## Output Format

### get — message list
```
#general — last 5 messages
──────────────────────────────────────────────────
[1712345678.001]  alice:    Hey team, standup in 5?
[1712345678.002]  bob:      On my way  (2 replies)
[1712345678.003]  carol:    Slides are up: https://...
```
Show `(N replies)` when reply_count > 0 — those are threads the user may want to drill into.

### reply — confirmation
```
✅ Sent to #general — ts: 1712345999.001
   Thread: 1712345678.002 (reply)
```

### error
```
❌ Error: Channel not found: #doesnotexist
```

## Notes

- Channel names can be `#name`, `name`, or a raw Slack ID (`C012AB3CD`).
- Thread timestamps (`ts`) are dot-decimal numbers like `1712345678.123456` — copy them from `get` output.
- The helper resolves channel names to IDs automatically by paginating `conversations.list`.
- Reading uses `user_token`; writing uses `bot_token`. If only one token is configured, operations requiring the missing token will fail with a clear message.
