import json
import subprocess
from pathlib import Path
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

_creds = json.loads((Path(__file__).parent.parent / "credentials" / "slack.json").read_text())

app = App(token=_creds["SLACK_BOT_TOKEN"])

@app.event("message")
def handle_dm_events(event, say, logger):
    # Only handle direct messages to the app
    if event.get("channel_type") != "im":
        return

    # Ignore bot messages to avoid loops
    if event.get("bot_id"):
        return

    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")

    logger.info(f"DM from user={user} channel={channel}: {text}")

    # Forward to Claude CLI and send response back
    result = subprocess.run(
        ["claude", "-p", text],
        capture_output=True,
        text=True,
    )
    response = result.stdout.strip() or result.stderr.strip() or "No response from Claude."
    logger.info(f"Claude response: {response[:100]}...")

    say(response)

if __name__ == "__main__":
    SocketModeHandler(app, _creds["SLACK_APP_TOKEN"]).start()