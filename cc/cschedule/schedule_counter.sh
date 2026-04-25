#!/bin/bash
COUNTER_FILE="/home/xingqianx/.claude/schedule_counter.txt"
COUNTER=$(cat "$COUNTER_FILE" 2>/dev/null || echo "0")
COUNTER=$((COUNTER + 1))
echo "$COUNTER" > "$COUNTER_FILE"
/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py \
  --owner-slack-id U0AT5LD6E9Y \
  --topic "schedule-counter" \
  --createtime "2026/04/25 00:40" \
  --message "schedule counter: $COUNTER"
