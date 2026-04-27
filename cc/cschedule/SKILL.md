---
name: cschedule
description: Set up a **repeating Claude query** tied to a launch. Agent mental model — cschedule schedules a query (a Claude prompt) that fires on a recurring cron schedule, until cancelled. Implementation — every cron line cschedule installs invokes `channels/slack_callback.py`, which dispatches the query to a fresh Claude session and exits. What the Claude session does after that, including any Slack posting via `slack.py`, is **out of cschedule's scope**. Use right after starting a long-running job/run/build to get periodic Claude-generated updates. NOT a general cron wrapper. Includes a `loop_runner.py` fallback when the host wipes user crontabs.
user_invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

## Purpose

**Agent mental model: cschedule sets up a repeating Claude query.** The user gives you (a) a question or instruction they want Claude to handle periodically and (b) a cadence; you install a cron entry that fires that query, repeatedly, until cancelled.

Concretely, the skill does **one thing**: install / list / cancel OS cron entries that fire `channels/slack_callback.py`. Every cron line cschedule installs is a slack_callback.py invocation — nothing else.

What slack_callback.py does on each fire:
1. Dispatches the `--message` argument — i.e. the query — to a fresh Claude session, with the identity arguments attached.
2. Exits.

That's all cschedule needs to know. **Slack posting is not cschedule's concern.** Whatever the Claude session does with the query — including any messaging via `slack.py` — is downstream of this skill. Do not document, modify, or reason about Slack-posting mechanics in cschedule.

Canonical workflow:
1. User launches a long-running thing (Slurm job, training run, build, deploy).
2. They invoke `/cschedule …` to set up a periodic query: cadence + the question Claude should handle each fire.
3. Cron fires the query on schedule.
4. When the launch is done, the user runs `/cschedule cancel <tag>`; the termination protocol calls `slack_callback.py --terminate` so the callback can clean up its own state.

**Out of scope.** Do not use cschedule for arbitrary periodic scripts, deferred reminders, standalone monitors, or any cron entry that does not call slack_callback.py. If a request doesn't fit the "repeating Claude query" pattern, push back — direct the user to `crontab -e` or another tool. The skill used to be generic; that scope was a mistake and has been removed.

The cron daemon ticks regardless of whether Claude is running — schedules survive session exit, harness restarts, and logout.

## PROHIBITED: the REPL `CronCreate` / `CronList` / `CronDelete` tools

**Do not call `CronCreate`, `CronList`, or `CronDelete`. They do not work in this environment. Do not attempt them, not even once, not even to "verify" — they are a dead end.**

Symptom: jobs appear to register, then silently disappear from `CronList` without ever firing. Past sessions have wasted significant time retrying these tools or trying to diagnose them. Don't.

This skill uses **OS cron exclusively** (Backend B), with the user-space loop runner (Backend C) as the only fallback. If you find yourself reaching for `CronCreate`, stop and re-read this section.

The cost of OS cron: Claude isn't in the loop on each fire. The cron line just hands off to slack_callback.py, which dispatches the query to a fresh Claude session — that's the contract.

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

Insert/update/delete operations preserve every other crontab line. The unique tag must be specific enough that no two schedules collide — include the launch being watched (e.g. `tag=watch-job-12345`, `tag=pulse-train-run-A`).

## Invocation

```
/cschedule every minute, run query "<prompt>"          # set up a recurring callback for the current launch
/cschedule list                                        # show all cschedule-managed entries
/cschedule cancel <tag>                                # delete one by tag (runs --terminate first)
/cschedule cancel-all                                  # delete every cschedule-managed entry
```

### Parsing the request

1. **Schedule** → cron expression in local time. Conventions:
   - `every N min` → `*/N * * * *` (use `*/N` only when `60 % N == 0`; otherwise pick an off-minute pattern like `7,22,37,52`)
   - `every N hours` → `0 */N * * *`
   - `every weekday at H:M` → `M H * * 1-5`
   - When the user is approximate ("around 9am", "hourly"), nudge the minute off `:00` and `:30` to spread load (e.g. `7 9 * * *` instead of `0 9 * * *`).
2. **Query** (`--message`) → the prompt Claude should run each fire. Always frame as an instruction or question.
3. **Tag** → a unique slug derived from the launch being watched. Default format: lowercase-hyphenated, ≤ 30 chars (e.g. `watch-job-12345`, `pulse-train-run-A`).
4. **Log path** → `/tmp/cschedule_<tag>.log`. Logs survive across fires; tail to debug.

## The slack_callback.py invocation

Every cschedule cron entry calls `channels/slack_callback.py`. From cschedule's perspective the callback does one thing: take a query plus identity arguments, dispatch to a Claude session, exit. Anything beyond that — Slack threads, anchors, @-mentions, message posting — is handled by `slack.py` and is **not** documented or managed here.

Schedule-time command:

```
/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py \
  --owner-slack-id <creator_slack_uid> \
  --topic <unique-topic> \
  --createtime <iso-datetime> \
  --message "<query for Claude>"
```

- **`--owner-slack-id`**: raw Slack user ID of the **person invoking `/cschedule`** (the creator). Resolve dynamically at install time via `cslack`; never resolve at fire time. Names, handles, and emails are rejected by the CLI validator — pass the raw UID.
- **`--topic`**: unique identifier for this schedule. Same `(owner, topic)` always resolves to the same downstream state — this is the idempotency key.
- **`--createtime`**: datetime string in the format `YYYY/mm/dd-HH:MM`, captured **at schedule time** (not fire time) and baked into the cron command. Combined with `(owner, topic)` this forms the full identity of the schedule. Use `$(date '+%Y/%m/%d %H:%M')` at install time.
- **`--message`**: the **query** (a Claude prompt) dispatched to a fresh Claude session on each fire. Write it as an instruction or question for Claude. Where Claude's response ends up is `slack.py`'s domain, not cschedule's.

## Operations

All operations below assume `bash` is available. Each is a single composable snippet you can paste into a Bash tool call.

### Install or update an entry by tag (idempotent)

```bash
TAG='<unique-tag>'
SCHEDULE='* * * * *'
OWNER='<creator slack UID, resolved via cslack at install time>'
TOPIC='<unique topic, often == TAG>'
CREATETIME=$(date '+%Y/%m/%d %H:%M')
PROMPT='<the Claude query for each fire>'
CMD="/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py --owner-slack-id ${OWNER} --topic ${TOPIC} --createtime '${CREATETIME}' --message '${PROMPT}'"
LOG="/tmp/cschedule_${TAG}.log"

(
  crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '
    $0 ~ t {skip=2; next}
    skip>0 {skip--; next}
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

**Important:** run `slack_callback.py --terminate` **before** removing the crontab line so the callback can clean up its own state. See "Termination protocol" below.

```bash
TAG='<unique-tag>'

LINE=$(crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '$0 ~ t {found=1; next} found {print; exit}')
PY_CMD=$(echo "$LINE" | sed -nE 's|.*(/usr/bin/python [^;}]*slack_callback\.py[^;}]*).*|\1|p')
if [ -n "$PY_CMD" ]; then
  eval "$PY_CMD --terminate"
fi

crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '
  $0 ~ t {skip=2; next}
  skip>0 {skip--; next}
  {print}
' | crontab -
```

### Cancel every cschedule-managed entry (leaves user's other cron lines intact)

Iterate so each entry gets its own termination call before its crontab block is removed:

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

### Termination protocol

Cancelling the crontab line alone is not enough — call `slack_callback.py --terminate` *before* removing the line so the callback can do its own cleanup (whatever downstream state it tracks; not cschedule's concern). Once the crontab line is gone, the `--owner-slack-id`, `--topic`, and `--createtime` triple is lost and the callback can no longer locate its state.

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

## Constraints

- **Never** install a cschedule cron entry that runs anything other than `channels/slack_callback.py`. Every line this skill installs must be a slack_callback.py invocation. If the user wants a non-callback cron, refuse and direct them to `crontab -e`.
- **Never** call `crontab <file>` with a file that doesn't preserve unrelated existing lines. Always read the current crontab first and merge — multiple Claude sessions may be managing different tags simultaneously.
- **Never** schedule a slack_callback.py entry without `--owner-slack-id <creator_user_id>`.
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
CMD='<slack_callback.py invocation>'

nohup /usr/bin/python /home/xingqianx/.claude/skills/cschedule/loop_runner.py \
  --tag "${TAG}" \
  --period-seconds ${PERIOD} \
  --cmd "${CMD}" \
  > /dev/null 2>&1 &

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

### Example 1 — repeating Claude query on a launch (the canonical case)

User: `/cschedule every minute, run query "Briefly check status of my Slurm jobs and report any state change."`

*(Typical context: the user just kicked off a long Slurm job / training run and wants Claude to periodically check on it. `--message` is the query Claude runs each fire; what Claude does with the answer is slack.py's job.)*

Agent:
1. Resolve inputs once at schedule time:
   - **Tag** = `watch-slurm-status`
   - **Schedule** = `* * * * *`
   - **Owner UID** = the current invoker's Slack UID, resolved dynamically via `cslack` from their email/username (do NOT hardcode). The resolved ID is then baked into the cron command.
   - **Createtime** = `$(date '+%Y/%m/%d %H:%M')` captured now
2. Install the cron line:
   ```bash
   TAG='watch-slurm-status'
   SCHEDULE='* * * * *'
   OWNER='<resolve via cslack from current user — e.g. U01ABCDEFGH>'
   TOPIC='watch-slurm-status'
   CREATETIME=$(date '+%Y/%m/%d %H:%M')
   PROMPT='Briefly check status of my Slurm jobs and report any state change.'
   CMD="/usr/bin/python /home/xingqianx/Project/trichord/channels/slack_callback.py --owner-slack-id ${OWNER} --topic ${TOPIC} --createtime '${CREATETIME}' --message '${PROMPT}'"
   LOG="/tmp/cschedule_${TAG}.log"
   (crontab -l 2>/dev/null | awk -v t="cschedule:tag=${TAG}" '$0 ~ t {skip=2; next} skip>0 {skip--; next} {print}'; \
    printf '# cschedule:tag=%s\n' "${TAG}"; \
    printf '%s { echo "--- $(/bin/date -Iseconds) cron-fire ---"; %s; } >> %s 2>&1\n' "${SCHEDULE}" "${CMD}" "${LOG}") | crontab -
   ```
3. Verify: `crontab -l | grep -A1 watch-slurm-status`.
4. Wait 75s, then `cat /tmp/cschedule_watch-slurm-status.log` — should show at least one cron fire entry.
5. Confirm to user with the standard report block.

### Example 2 — listing & cancelling

User: `/cschedule list`

Agent runs the list snippet:
```
watch-slurm-status | * * * * * { echo "--- $(...) cron-fire ---"; /usr/bin/python ... ; } >> /tmp/cschedule_watch-slurm-status.log 2>&1
```

User: `/cschedule cancel watch-slurm-status`

Agent runs the cancel-by-tag snippet (which calls `--terminate` first), then verifies `crontab -l` no longer contains the tag.
