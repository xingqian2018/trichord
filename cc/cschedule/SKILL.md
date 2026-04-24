---
name: cschedule
description: Schedule a callback prompt to fire in the SAME Claude session later — either once ("check in 5 min") or recurring ("every 15 min"). Wraps the built-in CronCreate tool and bakes in a Slack-posting convention so results surface in chat. Use for staged workflows (run stageB after stageA), polling (is the job done?), and deferred checks.
user_invocable: true
allowed-tools:
  - Bash
  - Read
---

## Purpose

Queue a prompt to re-enter *this same REPL session* at a future time. The prompt fires back into the same conversation — full context, same session ID — so the agent can continue where it left off and decide whether the thing it was waiting on has happened yet.

Typical uses:
- **Poll until done** — "in 5 min, run /checkrun and tell me if job 12345 finished"
- **Stage B after Stage A** — "every 10 min, check if the training is done; when yes, kick off eval"
- **Deferred follow-up** — "after market close, review today's trades in cinvest"
- **Monitor** — "every 30 min until I say stop, summarize new Slack messages in #alerts"

## How it works (under the hood)

This skill is a thin wrapper around the built-in **`CronCreate`** tool, which is session-scoped:

- Fires only while the REPL is idle (not mid-query).
- Lives only in THIS Claude session — when the REPL exits, queued jobs die. Tell the user this when scheduling.
- Recurring tasks auto-expire after 7 days.
- One-shot (`recurring: false`) fires once then deletes itself.

Because the callback re-enters the same REPL, the agent that fires **is** this same session — same memory, same context, same CLAUDE.md. No subprocess, no `claude -p --resume`, no session forking. That's intentional: simpler and keeps continuity.

## Invocation

The user invokes with a natural-language phrase:

```
/cschedule in 5 min, run /checkrun and tell me if job 12345 finished
/cschedule every 10 min, poll slurm until all my jobs are done, then post to #training
/cschedule at 4:05pm, review today's cinvest positions
/cschedule every weekday at 9:10am, run /checkrun
/cschedule list              # show active jobs
/cschedule cancel <job_id>   # delete a job
```

### Parse the request into:

1. **Schedule** — when to fire
   - One-shot delay: `in N min/hours` → compute an absolute cron for that minute, `recurring: false`
   - One-shot absolute time: `at HH:MM` → cron pinned to that minute today (or tomorrow if passed), `recurring: false`
   - Recurring: `every N min`, `hourly`, `weekdays at 9`, explicit cron expr → use the matching cron, `recurring: true`
2. **Callback prompt** — what to do when the cron fires
3. **Slack channel** (optional) — where to also post the result

### Compose the callback prompt

The prompt that gets queued into CronCreate should be **self-contained**: when it fires, the agent re-enters the session with full history but needs a clear instruction for *this* turn. Shape it like:

```
[scheduled callback — fired at <expected_time>]

Task: <what to do — e.g. "run /checkrun and determine whether job 12345 finished">

<if decision-making>
Decision: <e.g. "if the job is done, run `ssh awscode 'sbatch ~/jobs/eval.sh'` to kick off stage B; if still running, do nothing further this turn">
</if>

<if slack channel provided>
Report: after deciding, post a concise one-paragraph status to Slack channel <channel> via the cslack skill. Include: what you checked, what you found, what (if anything) you did next.
</if>
```

Keep the prompt tight — no re-stating context the session already has. But do name specific IDs, paths, flags, and decision rules so the callback doesn't have to guess.

### Call CronCreate

- **One-shot in N minutes from now**: compute `now + N min`, build cron `"<M> <H> <DoM> <Mon> *"` pinned to that absolute minute, set `recurring: false`. Avoid minute `0`/`30` only when the user's time isn't explicit — if they said "in 5 min" just use whatever minute you land on.
- **Recurring every N minutes**: `"*/N * * * *"` (pick an off-minute offset if N divides 60: e.g. every 15 min → `"7,22,37,52 * * * *"` instead of `"*/15 * * * *"` when the exact minute doesn't matter).
- **Absolute daily**: `"M H * * *"` — nudge M off `0`/`30` when the user was approximate.
- **Durable**: leave as default (false). Our callback prompts reference in-session context, so persisting beyond the session is meaningless.

### Confirm to the user

After creating, report:
```
✅ Scheduled: <one-line summary>
   Fires: <human time, e.g. "4:05pm today" or "every 10 min starting :07">
   Job id: <id from CronCreate>
   Note: lives in this REPL only — if you quit Claude, the job is lost.
```

If the user asked for Slack posting, mention the channel in the confirmation.

## Slack integration

When the user specifies a channel (e.g. "...and post to #training"), the **callback prompt** should instruct the agent to invoke `/cslack reply <channel> "<summary>"` (or call the `cslack` helper directly) after the check. The scheduler skill itself does NOT post to Slack at scheduling time — it only bakes the Slack instruction INTO the callback prompt, so posting happens when the callback fires and has a real result.

If no channel is given:
- For one-shots: the result just appears in the REPL when the callback fires.
- For recurring: consider asking the user whether they want Slack posting, since they may not be watching the REPL.

### @-mentioning the scheduler's creator

Scheduled Slack messages should @-mention the person who scheduled the callback — otherwise the message lands silently in the channel and is easy to miss. `slack_client.py reply` supports a `--mention` flag that accepts a raw user ID (`U01ABCDEFGH`), an `@username`, a username, or an email. It can be repeated to mention several people.

When composing a callback prompt that posts to Slack, prefer calling the helper directly so `--mention` is explicit:

```
python /home/xingqianx/.claude/skills/cslack/slack_client.py reply \
  --channel "#<channel>" \
  --mention "<creator_user_id_or_username>" \
  --text "<message>"
```

At schedule time, resolve the creator's Slack user ID once and bake it into the callback prompt (raw `U…` IDs are the most robust — they don't break if the person renames their Slack handle). If you don't have it handy, `users.list` via the Slack SDK returns everyone in the workspace; match on `name` or `real_name`. Email lookup needs the `users:read.email` scope, which most bot tokens don't have.

## Listing & cancelling

- **`/cschedule list`** → call `CronList` and render: id, schedule, next fire time, first ~80 chars of the prompt.
- **`/cschedule cancel <id>`** → call `CronDelete` with that id. If the user says "cancel all" or "cancel everything", list first, confirm, then delete each.

## Stop condition for recurring polls

When the user wants a poll that stops on success (the common case — "keep checking until the job is done"), the callback prompt MUST include a self-cancel instruction. Since the job id isn't known at prompt-composition time, use a **unique tag** the callback can self-find on:

```
[scheduled callback — tag:<unique-tag>]
...
When the condition is met, call CronList, find the job whose prompt contains
"tag:<unique-tag>", and CronDelete it. Then post the final "done" message.
```

Pick a tag that's specific enough to match only this job (include the thing being watched, e.g. `tag:watch-DAITR007`).

## Examples

### Example 1 — one-shot deferred check

User: `/cschedule in 7 min, run /checkrun and tell me if slurm job 84321 finished; post to #training if done`

Agent:
1. Now is 14:02 → fires at 14:09. Cron `"9 14 23 4 *"`, recurring=false.
2. Callback prompt:
   ```
   [scheduled callback — tag:check-84321]

   Task: run /checkrun and determine whether slurm job 84321 has completed
   (check both "COMPLETED" state and the log tail for errors).

   Report:
   - If the job is done, post a one-paragraph summary to Slack #training via
     `/cslack reply #training "<summary>"`. Include job id, outcome, log path.
   - If still running, just note "still running" and do not post to Slack.
   ```
3. Calls `CronCreate(cron="9 14 23 4 *", prompt=<above>, recurring=false)`.
4. Confirms: `✅ Scheduled check at 14:09 — job id abc123. Lives in this REPL.`

### Example 2 — recurring with self-cancel

User: `/cschedule every 15 min, check if training job DAITR007 is done, and when it is, kick off eval and tell me in #training`

Agent:
1. Cron `"7,22,37,52 * * * *"`, recurring=true.
2. Callback prompt:
   ```
   [scheduled callback — tag:watch-DAITR007]

   Task: SSH to awscode, run `squeue -u xingqianx -h -o '%i %j %T' | grep DAITR007`
   to see if the training job is still in the queue. If absent, check the most
   recent log in `~/log/slurm/*DAITR007*.o` for "COMPLETED" or error markers.

   Decisions:
   - Still running → take no action, stay silent.
   - Failed → post a failure summary to Slack #training AND self-cancel
     (CronList → find job with prompt containing "tag:watch-DAITR007" → CronDelete).
   - Completed successfully → submit eval via
     `ssh awscode 'cd ~/Project/... && slaunch small 1 eval_DAITR007 <script>'`,
     post kickoff confirmation to #training, AND self-cancel.
   ```
3. `CronCreate(cron="7,22,37,52 * * * *", prompt=<above>, recurring=true)`.
4. Confirms: `✅ Watching DAITR007 every 15 min (:07,:22,:37,:52). Will self-stop on completion or failure. Job id xyz789.`

### Example 3 — listing

User: `/cschedule list`

Agent: calls `CronList`, renders:
```
Active schedules (2):
  [abc123]  one-shot @ 14:09       "check slurm job 84321…"
  [xyz789]  */15 min (7,22,37,52)  "watch-DAITR007 training…"
```

## Constraints

- **Never** use OS-level crontab / `at` / background sleep for this skill — that's a different scheduling model. This skill is strictly session-scoped via `CronCreate`.
- **Never** schedule a job without making the callback prompt self-contained enough to execute without the user present.
- **Never** let a recurring poll run forever without a self-cancel condition for the success/failure case — if the user didn't give one, ask.
- The 7-day auto-expiry on recurring jobs is a built-in CronCreate behavior — surface it to the user when they schedule long-lived polls.
- If the user quits the REPL, all schedules die. Warn on any job expected to fire more than a few hours out.
