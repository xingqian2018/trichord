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

Config lives at `~/.claude/skills/cslack/config.json`:

```json
{
  "bot_token":       "xoxb-...",
  "user_token":      "xoxp-...",
  "default_channel": "#your-channel"
}
```

- **`bot_token`** — required for sending (posting messages). Create a Slack app, add `chat:write` scope, install to workspace, copy Bot OAuth token.
- **`user_token`** — required for reading channel history (`conversations:history`, `conversations:replies`). Needed because bot tokens may lack history read access depending on Slack plan.
- **`default_channel`** — optional fallback when no channel is specified.

If config is missing, the client prints a clear error with the path.

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
