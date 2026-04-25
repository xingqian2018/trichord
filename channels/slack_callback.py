#!/usr/bin/env python3
import argparse
import fcntl
import json
import sys
from pathlib import Path

from slack_sdk import WebClient

CREDS = json.loads((Path(__file__).resolve().parent.parent / "credentials" / "slack.json").read_text())

CHANNELS_TMP = Path(__file__).resolve().parent / "tmp"

SESSION_MAPPING_FILE = CHANNELS_TMP / "slack_thread_to_claude_session_mapping.json"
SESSION_MAPPING_LOCKFILE = CHANNELS_TMP / "slack_thread_to_claude_session_mapping.lock"

TOPIC_INDEX_FILE = CHANNELS_TMP / "cschedule_topic_index.json"
TOPIC_INDEX_LOCKFILE = CHANNELS_TMP / "cschedule_topic_index.lock"

slack = WebClient(token=CREDS["SLACK_BOT_TOKEN"])


def load_locked(path: Path, lock_path: Path) -> dict:
    if not path.exists():
        return {}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_SH)
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}


def update_locked(path: Path, lock_path: Path, key: str, value) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        data: dict = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}
        if data.get(key) == value:
            return
        data[key] = value
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)


def delete_locked(path: Path, lock_path: Path, key: str) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return
        if key not in data:
            return
        data.pop(key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(path)


def topic_key(owner_slack_id: str, topic: str, createtime: str) -> str:
    return f"{owner_slack_id}|{topic}|{createtime}"


def find_thread(topic_index: dict, owner_slack_id: str, topic: str, createtime: str):
    return topic_index.get(topic_key(owner_slack_id, topic, createtime))


def post(channel: str, text: str, *, thread_ts: str | None = None) -> dict:
    kwargs = {"channel": channel, "text": text}
    if thread_ts:
        kwargs["thread_ts"] = thread_ts
    try:
        resp = slack.chat_postMessage(**kwargs)
        return {"success": True, "channel_id": resp["channel"], "ts": resp["ts"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update(channel: str, ts: str, text: str) -> dict:
    try:
        resp = slack.chat_update(channel=channel, ts=ts, text=text)
        return {"success": True, "channel_id": resp["channel"], "ts": resp["ts"]}
    except Exception as e:
        return {"success": False, "error": str(e)}


def slack_user_id(s: str) -> str:
    if not (len(s) >= 9 and s[0] in ("U", "W") and s.isalnum()):
        raise argparse.ArgumentTypeError(f"must be a Slack user ID (e.g. U01ABCDEFGH), got {s!r}")
    return s


def render_anchor(owner_slack_id: str, topic: str, now_iso: str, status: str) -> str:
    badge = ":large_green_circle: Live" if status == "running" else ":red_circle: Closed"
    return f":thread: <@{owner_slack_id}> *{topic}* {now_iso}  {badge}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--owner-slack-id", required=True, type=slack_user_id)
    p.add_argument("--topic", required=True)
    p.add_argument("--createtime", required=True)
    p.add_argument("--message", required=True)
    p.add_argument("--terminate", action="store_true", help="Final run: post message, mark anchor closed, drop topic-index entry")
    args = p.parse_args()

    topic_index = load_locked(TOPIC_INDEX_FILE, TOPIC_INDEX_LOCKFILE)
    existing = find_thread(topic_index, args.owner_slack_id, args.topic, args.createtime)
    tkey = topic_key(args.owner_slack_id, args.topic, args.createtime)

    if args.terminate:
        if existing is None:
            print(json.dumps({"stage": "terminate", "success": False, "error": "no existing thread"}, indent=2))
            return 2

        anchor_text = render_anchor(args.owner_slack_id, args.topic, args.createtime, "closed")
        anchor_update = update(existing["channel_id"], existing["thread_ts"], anchor_text)
        delete_locked(TOPIC_INDEX_FILE, TOPIC_INDEX_LOCKFILE, tkey)

        print(json.dumps(
            {
                "stage": "terminated",
                "thread_key": existing["thread_key"],
                "anchor_updated": anchor_update.get("success"),
                "topic_index_cleared": True,
                "success": bool(anchor_update.get("success")),
            },
            indent=2,
        ))
        return 0 if anchor_update.get("success") else 2

    if existing is None:
        anchor_text = render_anchor(args.owner_slack_id, args.topic, args.createtime, "running")

        anchor = post(CREDS["CALLBACK_CHANNEL"], anchor_text)
        if not anchor.get("success"):
            print(json.dumps({"stage": "anchor", **anchor}, indent=2))
            return 2
        channel_id = anchor["channel_id"]
        thread_ts = anchor["ts"]
        thread_key = f"{channel_id}:{thread_ts}"

        entry = {
            "thread_key": thread_key,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "owner_slack_id": args.owner_slack_id,
            "topic": args.topic,
            "status": "running",
            "created_at": args.createtime,
        }
        update_locked(TOPIC_INDEX_FILE, TOPIC_INDEX_LOCKFILE, tkey, entry)

        session_mapping = load_locked(SESSION_MAPPING_FILE, SESSION_MAPPING_LOCKFILE)
        session_id = session_mapping.get(thread_key)

        msg = post(channel_id, args.message, thread_ts=thread_ts)
        print(json.dumps(
            {
                "stage": "created",
                "thread_key": thread_key,
                "anchor_ts": thread_ts,
                "message_ts": msg.get("ts"),
                "session_id": session_id,
                "success": bool(msg.get("success")),
            },
            indent=2,
        ))
        return 0 if msg.get("success") else 2

    thread_key = existing["thread_key"]
    session_mapping = load_locked(SESSION_MAPPING_FILE, SESSION_MAPPING_LOCKFILE)
    session_id = session_mapping.get(thread_key)

    reply = post(existing["channel_id"], args.message, thread_ts=existing["thread_ts"])
    print(json.dumps(
        {
            "stage": "reply",
            "thread_key": thread_key,
            "message_ts": reply.get("ts"),
            "session_id": session_id,
            "success": bool(reply.get("success")),
            "error": reply.get("error"),
        },
        indent=2,
    ))
    return 0 if reply.get("success") else 2


if __name__ == "__main__":
    sys.exit(main())
