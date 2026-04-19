import json
from pathlib import Path
import anthropic
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

CREDS = json.loads((Path(__file__).parent.parent / "credentials" / "slack.json").read_text())

BOT_USER_ID = None

CLAUDE = None


def get_claude():
    global CLAUDE
    if CLAUDE is None:
        creds = json.loads((Path(__file__).parent.parent / "credentials" / "claude.json").read_text())
        CLAUDE = anthropic.Anthropic(api_key=creds["primaryApiKey"])
    return CLAUDE

app = App(token=CREDS["SLACK_BOT_TOKEN"])

def get_bot_user_id():
    global BOT_USER_ID
    if BOT_USER_ID is None:
        BOT_USER_ID = app.client.auth_test()["user_id"]
    return BOT_USER_ID


def build_messages(event, channel):
    thread_ts = event.get("thread_ts")
    current_ts = event.get("ts")
    text = event.get("text", "")

    if not thread_ts or thread_ts == current_ts:
        return [{"role": "user", "content": text}]

    bot_uid = get_bot_user_id()
    result = app.client.conversations_replies(channel=channel, ts=thread_ts)
    messages = []

    for msg in result.get("messages", []):
        if msg.get("ts") == current_ts:
            continue
        is_bot = msg.get("bot_id") or msg.get("user") == bot_uid
        role = "assistant" if is_bot else "user"
        content = msg.get("text", "")
        if not content:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + content
        else:
            messages.append({"role": role, "content": content})

    while messages and messages[0]["role"] == "assistant":
        messages.pop(0)

    messages.append({"role": "user", "content": text})
    return messages


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

    messages = build_messages(event, channel)
    response = get_claude().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=messages,
    )
    reply = next(b.text for b in response.content if b.type == "text")
    logger.info(f"Claude response: {reply[:30]}")
    say(text=reply, thread_ts=thread_ts)


if __name__ == "__main__":
    SocketModeHandler(app, CREDS["SLACK_APP_TOKEN"]).start()
