---
name: cschedule
description: Schedule a shell command to fire on a recurring or one-shot schedule, with safe per-entry add/list/cancel by tag and a built-in Slack-posting + @-mention-creator convention. Two backends — OS crontab (default) and a self-contained Python loop runner (`loop_runner.py`) used as a fallback when the host periodically wipes user crontabs or has no cron daemon. Use for recurring Slack pings, periodic script runs, deferred reminders — anything where reliability matters more than having Claude reasoning per fire. Schedules survive Claude session exit and logout regardless of backend.
user_invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

## Purpose

Add or remove entries in the user's **OS crontab** so a shell command fires on a real schedule. The cron daemon ticks regardless of whether Claude is running — schedules survive session exit, harness restarts, and logout. This is the reliable scheduling backend.

Typical uses:
- **Recurring Slack ping** — "every minute, post X to #channel"
- **Periodic script** — "every weekday at 9:10am, run ~/bin/standup.sh"
- **Deferred reminder** — "in 30 min, post 'check the deploy' to #ops"
- **Long-running monitor** — "every 15 min, run my_script.py and Slack the result if it changed"

## Why not the REPL `CronCreate` tool?

`CronCreate` claims to fire callbacks back into the same Claude session, which would let the agent reason about each fire. In practice, this harness has been observed to silently drop those jobs — they register, then disappear from `CronList` without ever firing visibly. **Do not rely on `CronCreate` for anything that needs to actually happen.** This skill uses OS cron exclusively.

The cost: Claude isn't in the loop on each fire. Whatever the cron does has to be a self-contained shell/python invocation. If the user wants "look at output, decide, do next thing", encode that decision logic in a shell or python script the cron runs — not in a Claude prompt.

## Two backends, picked by environment

This skill ships two execution backends. Always **try crontab first**; fall back to the loop runner only when crontab is observably unreliable in the current environment.

| Backend | Mechanism | When to use |
|---|---|---|
| **B — OS crontab** | Adds a tagged line to the user's `crontab`; the system cron daemon fires it. | Default. Simplest, best when the host honors user crontabs. |
| **C — Loop runner** | A long-running `nohup`'d Python process (`loop_runner.py`) that ticks on its own internal clock. | When Backend B is being wiped, when the host has no cron daemon, or when the schedule needs to outlive a session-scoped sandbox. |

**How to know which environment you're in:** install a one-minute crontab entry with a log file, wait 2–3 minute boundaries past install, then check:

- If `crontab -l` still shows the entry **and** the log has entries for every minute since install → Backend B is healthy. Stop here.
- If the entry vanishes from `crontab -l` between fires (even though the daemon was running) → the host periodically wipes user crontabs. Use Backend C.
- If `crontab -l` shows the entry but the log is empty after 2 boundaries → cron daemon isn't honoring user crontabs (less common). Use Backend C.

Backend C is documented in detail further below ("Backend C — loop runner").

## Concurrency safety — tag-based management

Multiple Claude sessions can run at the same time, all wanting to manage the same user's crontab. `crontab <file>` replaces the entire crontab and clobbers other sessions' entries.

**Always use tag-based per-block edits.** Each managed entry is two lines:

```
# cschedule:tag=<unique-tag>
<schedule> <command>
```

Insert/update/delete operations preserve every other crontab line. The unique tag must be specific enough that no two schedules collide — include the action being scheduled (e.g. `tag=ping-good-gsb`, `tag=watch-DAITR007`, `tag=standup-9am`).

## Invocation

```
/cschedule every minute, post "Good" to #gsb_scheduled_event
/cschedule every weekday at 9:10am, run ~/bin/standup.sh
/cschedule in 30 min, post "check the deploy" to #ops
/cschedule list                 # show all cschedule-managed entries
/cschedule cancel <tag>         # delete one by tag
/cschedule cancel-all           # delete every cschedule-managed entry
```

### Parsing the request

1. **Schedule** → cron expression in local time. Conventions:
   - `every N min` → `*/N * * * *` (use `*/N` only when `60 % N == 0`; otherwise pick an off-minute pattern like `7,22,37,52`)
   - `every N hours` → `0 */N * * *`
   - `every weekday at H:M` → `M H * * 1-5`
   - `at H:M` (one-shot) → see "One-shots" below
   - When the user is approximate ("around 9am", "hourly"), nudge the minute off `:00` and `:30` to spread load (e.g. `7 9 * * *` instead of `0 9 * * *`).
2. **Command** → what to run. For Slack posts, use the helper (see "Slack convention"). For arbitrary scripts, use absolute paths and `/usr/bin/python` (cron's PATH is minimal).
3. **Tag** → a unique slug derived from the action. Default format: lowercase-hyphenated, ≤ 30 chars (e.g. `ping-good-gsb`, `standup-weekday-9am`).
4. **Log path** → `/tmp/cschedule_<tag>.log`. Logs survive across fires; tail to debug.

## Slack convention — use `channels/slack_callback.py`

Every Slack-posting cschedule entry should go through `channels/slack_callback.py` (in the user's trichord project), **not** `slack_client.py reply` directly. The callback wraps `slack_client.py` and adds:

1. **Thread-per-schedule**: the first fire creates one anchor message in the channel, and every subsequent fire posts as a reply in that same thread. This keeps the channel tidy instead of flooding it with N top-level messages.

All other state (thread lookup by `(owner, topic)`, routing replies back to a Claude session) is managed internally by the callback — cschedule doesn't need to know about those files.

The posting channel is not a CLI flag — it's read from `credentials/slack.json` at import time.

Schedule-time command:

```
/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py \
  --owner-slack-id <creator_slack_uid> \
  --topic <unique-topic> \
  --createtime <iso-datetime> \
  --message "<message>"
```

- **`--owner-slack-id`**: raw Slack user ID of the **person invoking `/cschedule`** (the creator) — names, handles, and emails are rejected by the CLI validator. Resolve it dynamically at install time via `cslack` (`users.lookupByEmail` requires the bot's OAuth to include `users:read.email`); never resolve at fire time. The `<@UID>` render only works with real IDs — Slack will not resolve plain `@handle` text sent via the API.
- **`--topic`**: unique identifier for this schedule. Same `(owner, topic)` always resolves to the same thread — this is the idempotency key.
- **`--createtime`**: datetime string in the format `YYYY/mm/dd HH:MM`, captured **at schedule time** (not fire time) and baked into the cron command. Combined with `(owner, topic)` this forms the full identity of the schedule — same triple always resolves to the same thread; a different createtime is treated as a different schedule. Use `$(date '+%Y/%m/%d %H:%M')` at install time.
- **`--message`**: the text to post on this fire.

## Operations

All operations below assume `bash` is available. Each is a single composable snippet you can paste into a Bash tool call.

### Install or update an entry by tag (idempotent)

```bash
TAG='<unique-tag>'                 # e.g. ping-good-gsb
SCHEDULE='* * * * *'               # any valid cron expression
CMD='<command>'                    # full command, single-quote-safe
LOG="/tmp/cschedule_${TAG}.log"

(
  crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '
    $0 ~ t {skip=2; next}          # drop the comment line
    skip>0 {skip--; next}          # and the cron line directly below
    {print}
  '
  printf '# cschedule:tag=%s\n' "${TAG}"
  printf '%s { echo "--- $(/bin/date -Iseconds) cron-fire ---"; %s; } >> %s 2>&1\n' \
    "${SCHEDULE}" "${CMD}" "${LOG}"
) | crontab -
```

Then verify: `crontab -l` should show the new tag block. After ~75 seconds, `cat /tmp/cschedule_<tag>.log` should show at least one fire (for sub-minute schedules).

### List all cschedule-managed entries

```bash
crontab -l 2>/dev/null | awk '
  /^# cschedule:tag=/ {
    tag=$0; sub(/^# cschedule:tag=/, "", tag)
    getline cmd
    print tag " | " cmd
  }
'
```

### Cancel one entry by tag

**Important:** if the entry posts to Slack via `slack_callback.py`, run a final `--terminate` invocation **before** removing the crontab line. This closes the thread on the Slack side: it edits the anchor message badge from "Live" to "Closed" and clears the topic-index entry. See the "Termination protocol for Slack-callback schedules" section below.

```bash
TAG='<unique-tag>'

# 1. Pull the existing slack_callback.py invocation out of the cron line and
#    re-run it verbatim with --terminate appended (same owner/topic/createtime/message).
LINE=$(crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '$0 ~ t {found=1; next} found {print; exit}')
PY_CMD=$(echo "$LINE" | sed -nE 's|.*(/usr/bin/python [^;}]*slack_callback\.py[^;}]*).*|\1|p')
if [ -n "$PY_CMD" ]; then
  eval "$PY_CMD --terminate"
fi

# 2. Remove the crontab block.
crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '
  $0 ~ t {skip=2; next}
  skip>0 {skip--; next}
  {print}
' | crontab -
```

### Cancel every cschedule-managed entry (leaves user's other cron lines intact)

Iterate so each Slack-callback entry gets its own termination ping before its crontab block is removed:

```bash
for TAG in $(crontab -l 2>/dev/null | awk '/^# cschedule:tag=/ {sub(/^# cschedule:tag=/, ""); print}'); do
  LINE=$(crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '$0 ~ t {found=1; next} found {print; exit}')
  PY_CMD=$(echo "$LINE" | sed -nE 's|.*(/usr/bin/python [^;}]*slack_callback\.py[^;}]*).*|\1|p')
  if [ -n "$PY_CMD" ]; then
    eval "$PY_CMD --terminate"
  fi
done

crontab -l 2>/dev/null | awk '
  /^# cschedule:tag=/ {skip=2; next}
  skip>0 {skip--; next}
  {print}
' | crontab -
```

### Termination protocol for Slack-callback schedules

Schedules that post via `slack_callback.py` have **state on the Slack side** that outlives the crontab entry: an anchor message in the channel and an entry in `cschedule_topic_index.json`. Cancelling the cron without telling the callback leaves the anchor showing "Live" forever and leaves a stale topic-index entry that will collide if the same `(owner, topic, createtime)` triple is ever reused.

`slack_callback.py --terminate` is the cleanup hook:
- (a) Posts the final `--message` into the existing thread.
- (b) Edits the anchor message — green-circle "Live" → red-circle "Closed".
- (c) Removes the topic-index entry.

Always run terminate **before** removing the crontab line — once the line is gone, you've lost `--owner-slack-id`, `--topic`, and `--createtime`, and the callback can no longer find the thread.

## One-shots

OS cron doesn't natively express "fire once". Pick:

1. **`at`** if available — `echo '<cmd>' | at HH:MM` is the cleanest. Check `command -v at` first; on minimal containers it isn't always installed.
2. **Self-deleting cron line** — date-pin the cron expression (e.g. `9 14 23 4 *` for "April 23 at 14:09") and append `&& <cancel-by-tag-snippet>` to the command so it removes itself after firing successfully. Heavier but works without `at`.

Prefer `at` when available. When using cron-with-self-delete, the tag must be unique because the self-delete uses the same tag-based awk filter.

## Health checks

Before reporting "scheduled" to the user, confirm:

1. `crontab -l` shows the new tag block.
2. `pgrep -a cron` returns a running daemon (e.g. `/usr/sbin/cron -f`).
3. For sub-minute schedules: wait one minute boundary past install time, then `cat /tmp/cschedule_<tag>.log`. If empty, something's wrong (check `/var/log/syslog` for cron errors).

If the cron daemon isn't running, this skill cannot work — surface that immediately rather than reporting a false success.

## Confirm to the user

After install, report:
```
✅ Scheduled: <one-line summary>
   Tag: <unique-tag>
   Schedule: <cron expr> (<human time, e.g. "every minute" or "weekdays at 9:10am">)
   Logs: /tmp/cschedule_<tag>.log
   Cancel: /cschedule cancel <tag>   (or:  crontab -e   to inspect)
   Note: persists across Claude sessions and logout — actively running on the OS.
```

For Slack-posting entries, also confirm: target channel, @-mentioned user, and that the bot has been invited to the channel (otherwise posts will fail with `not_in_channel`).

## Constraints

- **Never** call `crontab <file>` with a file that doesn't preserve unrelated existing lines. Always read the current crontab first and merge — multiple Claude sessions may be managing different tags simultaneously.
- **Never** schedule a Slack-posting entry without `--owner-slack-id <creator_user_id>` so the anchor message mentions someone.
- **Never** resolve a user at fire time when the schedule could resolve it once at install time. Cron-run shells have minimal env; bake fully-resolved IDs into the command.
- **Never** assume cron's PATH includes anything beyond `/usr/bin:/bin`. Use absolute paths for `python`, custom binaries, and any script invocation.
- **Never** report "scheduled" without verifying with `crontab -l`. Some environments restrict `crontab` writes — surface that as an error, not a success.

## Backend C — loop runner

When OS crontab isn't reliable, fall back to `loop_runner.py` — a self-contained Python script in this skill's folder that sleeps to wall-clock period boundaries and runs a shell command on each tick. It's not a daemon framework, just a single-file `while True: align-and-run` loop with a flock-based singleton guard.

### Why this exists

Some sandboxed environments wipe `/var/spool/cron/crontabs/` periodically (every ~minute) even though the cron daemon is running. The wipe happens after the daemon already fired any pending entries, so you see one or two fires and then silence — and `crontab -l` reads back empty even though you just installed it. A user-space loop process bypasses the wipe entirely because it doesn't depend on `/var/spool/cron`.

### Spawn an entry

```bash
TAG='<unique-tag>'
PERIOD=60                    # seconds between fires; aligns to wall-clock
CMD='<command>'              # passed to /bin/sh -c on each fire

nohup /usr/bin/python /home/xingqianx/.claude/skills/cschedule/loop_runner.py \
  --tag "${TAG}" \
  --period-seconds ${PERIOD} \
  --cmd "${CMD}" \
  > /dev/null 2>&1 &

# Verify it's alive
sleep 1
ps -fp "$(cat /tmp/cschedule_${TAG}.pid)"
```

The runner:
- Acquires an exclusive flock on `/tmp/cschedule_<tag>.pid` — re-spawning with the same tag is a no-op (idempotent).
- Logs every fire (with a `(tag=<tag>)` marker) to `/tmp/cschedule_<tag>.log`.
- Aligns to wall-clock multiples of the period: at `--period-seconds 60`, fires at `:00` of each minute, regardless of when it was started.

### List loop-runner entries

```bash
for pidfile in /tmp/cschedule_*.pid; do
  [ -e "$pidfile" ] || continue
  tag=$(basename "$pidfile" .pid)
  tag=${tag#cschedule_}
  pid=$(cat "$pidfile" 2>/dev/null)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    cmd=$(ps -p "$pid" -o cmd= 2>/dev/null)
    echo "$tag (pid $pid) | $cmd"
  fi
done
```

### Cancel one entry by tag

```bash
TAG='<unique-tag>'
pid=$(cat /tmp/cschedule_${TAG}.pid 2>/dev/null)
[ -n "$pid" ] && kill "$pid" && rm -f /tmp/cschedule_${TAG}.pid
```

### Cancel every loop-runner entry

```bash
for pidfile in /tmp/cschedule_*.pid; do
  [ -e "$pidfile" ] || continue
  pid=$(cat "$pidfile" 2>/dev/null)
  [ -n "$pid" ] && kill "$pid" 2>/dev/null
  rm -f "$pidfile"
done
```

### Caveats

- **No restart-on-crash.** If the Python process dies (OOM, kernel kill), nothing brings it back. For critical schedules, wrap the spawn in a watchdog or use systemd-user units instead.
- **Dies on host reboot.** Survives Claude session exit and shell logout, but not power cycles. Re-spawn from `~/.bashrc` or a systemd-user unit if cross-reboot persistence matters.
- **No sub-second precision.** The runner aligns to wall-clock multiples of `--period-seconds`, so cumulative drift stays bounded, but a single fire can be a few hundred ms late under load.

## Examples

### Example 1 — recurring Slack ping (the canonical case)

User: `/cschedule every minute, post "Good" to #gsb_scheduled_event`

Agent:
1. Resolve inputs once at schedule time:
   - **Tag** = `ping-good-gsb`
   - **Schedule** = `* * * * *`
   - **Owner UID** = the current invoker's Slack UID, resolved dynamically via `cslack` from their email/username (do NOT hardcode). The resolved ID is then baked into the cron command.
   - **Createtime** = `$(date '+%Y/%m/%d %H:%M')` captured now
2. Install the cron line that calls `slack_callback.py` (which handles thread creation on first fire, thread-reply on subsequent fires):
   ```bash
   TAG='ping-good-gsb'
   SCHEDULE='* * * * *'
   OWNER='<resolve via cslack from current user — e.g. U01ABCDEFGH>'
   TOPIC='ping-good-gsb'
   CREATETIME=$(date '+%Y/%m/%d %H:%M')
   CMD="/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py --owner-slack-id ${OWNER} --topic ${TOPIC} --createtime '${CREATETIME}' --message 'Good'"
   LOG="/tmp/cschedule_${TAG}.log"
   (crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '$0 ~ t {skip=2; next} skip>0 {skip--; next} {print}'; \
    printf '# cschedule:tag=%s\n' "${TAG}"; \
    printf '%s { echo "--- $(/bin/date -Iseconds) cron-fire ---"; %s; } >> %s 2>&1\n' "${SCHEDULE}" "${CMD}" "${LOG}") | crontab -
   ```
3. Verify: `crontab -l | grep -A1 ping-good-gsb`.
4. Wait 75s, then `cat /tmp/cschedule_ping-good-gsb.log` — first entry should be `"stage": "created"`, subsequent entries `"stage": "reply"`.
5. Confirm to user with the standard report block, plus the Slack thread link from the callback's first-fire output so they can jump straight to the thread.

### Example 2 — weekday standup script

User: `/cschedule every weekday at 9:07am, run ~/bin/standup.sh and Slack the output to #standup`

Agent:
1. Tag: `standup-weekday`. Cron: `7 9 * * 1-5`. The script + Slack post composes nicely as one shell command.
2. Install with:
   ```
   CMD='/home/xingqianx/bin/standup.sh 2>&1 | head -c 3000 | xargs -0 -I{} /usr/bin/python /home/xingqianx/.claude/skills/cslack/slack_client.py reply --channel "#standup" --mention "U0AT5LD6E9Y" --text {}'
   ```
3. (Note: complex pipelines like the above benefit from a tiny wrapper script — write it to `~/bin/standup_and_post.sh` and have cron run that, instead of cramming everything into the crontab line.)

### Example 3 — listing & cancelling

User: `/cschedule list`

Agent runs the list snippet:
```
ping-good-gsb | * * * * * { echo "--- $(...) cron-fire ---"; /usr/bin/python ... ; } >> /tmp/cschedule_ping-good-gsb.log 2>&1
standup-weekday | 7 9 * * 1-5 { echo "--- ..." ...
```

User: `/cschedule cancel ping-good-gsb`

Agent runs the cancel-by-tag snippet, then verifies `crontab -l` no longer contains the tag.
