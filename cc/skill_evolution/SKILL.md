---
name: skill_evolution
description: Edit a skill's SKILL.md file — either by applying explicit user instructions, or by extracting learnings from the current conversation. Does NOT invoke the target skill. Invoke as /skill_evolution <skill_name> [instruction].
user_invocable: true
---

# skill_evolution

Read and rewrite `~/.claude/skills/<skill_name>/SKILL.md`. This skill only edits the instruction file — it does **not** execute or invoke the target skill.

## Input forms

```
/skill_evolution <skill_name>
/skill_evolution <skill_name> <instruction>
```

- `<skill_name>` — directory name under `~/.claude/skills/` (e.g. `cinvest`, `ccode`, `checkrun`)
- `<instruction>` — optional explicit edit directive (e.g. "remove the confirmation step", "add a rule to never delete files")

If no skill name is given, ask the user which skill to target before proceeding.

## Step 1 — Read the skill file

Resolve: `~/.claude/skills/<skill_name>/SKILL.md`

Read the full file. If it does not exist, tell the user and stop.

## Step 2 — Determine mode

### Mode A — Explicit instruction (instruction was provided)

Apply exactly what the user asked. Interpret the instruction literally against the current file content:
- Additions: insert in the most natural location
- Removals: delete only what was specified
- Rewrites: change only the targeted section
- If the instruction is ambiguous, ask one clarifying question before drafting

### Mode B — Dialog extraction (no instruction given)

Scan the current conversation for **durable, non-obvious learnings** not already in the file:

- **Corrections** — user said "no", "not that", "stop doing X", redirected the approach
- **New patterns** — a workflow or heuristic that emerged and succeeded
- **Edge cases** — behavior that should differ under specific conditions
- **Constraints** — things that must always or never be done
- **Confirmed approaches** — non-obvious choices accepted without pushback

Skip: things already in the file, ephemeral task details (specific filenames, one-off values, current job IDs), generic best practices not specific to this skill.

If nothing meaningful is found, tell the user "no new learnings to add" and stop.

## Step 3 — Draft the update

Apply the changes to the file content:
- Preserve all unaffected content and structure
- Match the existing tone and style
- Do not add sections unless necessary

## Step 4 — Show and confirm

Present the proposed changes:
- **Mode A**: describe what was changed and where
- **Mode B**: list each change with a one-line rationale (which part of the dialog drove it)

Ask: "Apply these changes?" and wait for confirmation before writing.

If the user wants adjustments, revise and re-confirm.

## Step 5 — Write

Once confirmed, overwrite `~/.claude/skills/<skill_name>/SKILL.md`.

Report: "Updated `<skill_name>/SKILL.md` with N change(s)."
