---
name: nvtop5
description: Draft Xingqian's monthly "top 5" accomplishments email. Reads prior emails from this skill's `top5_log/` folder to mirror tone/structure, takes the user's notes for the latest month, and writes the new draft to `top5_log/top5_<YYYYMMDD>.md`. Invoke as /nvtop5.
user_invocable: true
---

# nvtop5

Draft a monthly top-5 accomplishments email in the same style as prior months.

## Inputs

- **History** — every `*.md` file in `top5_log/` is a prior monthly email the user has already converted to markdown. Treat these as the source of truth for subject, greeting, section headings, bullet shape, link/metric density, sign-off, and recipients.
- **This month's notes** — free-form bullets / snippets / project mentions the user supplies after invoking. May include slurm jobs, PRs, doc links, run names, etc.
- **(Optional) reply context** — if the user pastes a current email thread (e.g. a request from manager) along with the invocation, use it to set Subject and recipients and to honor any prompts/headings it asks for.

## Steps

1. **Read the history.** List `top5_log/*.md` and read the 1–3 most recent files (filenames are date-stamped, so sorted = chronological). Identify the recurring shape:
   - subject line pattern (substitute the new month at draft time)
   - greeting and recipient style
   - section ordering and heading wording
   - per-bullet length, whether links/metrics/run names are inline
   - closing line and signature
2. **If `top5_log/` is empty**, stop and ask the user to seed at least one prior example. Do not invent a format.
3. **Collect this month's input.** If the user did not paste notes with the invocation, ask:
   > Paste your notes / accomplishments for `<Month YYYY>` — raw bullets are fine, I'll shape them to match the historical format.
4. **Draft the email.** Match the latest historical entry exactly in structure and tone. Default to 5 items unless the history clearly varies. Do not invent accomplishments — if notes are thin, ask one clarifying round first.
5. **Write the output file.** Path: `top5_log/top5_<YYYYMMDD>.md` where `<YYYYMMDD>` is today's date. If a file with that name already exists, suffix `_v2`, `_v3`, … rather than overwriting.
6. **Display the draft in chat** as a fenced markdown block so the user can copy it, and report the saved file path as a clickable markdown link.

## Out of scope

- Does **not** send email — drafting only.
- Does **not** auto-convert old emails into markdown. Curating `top5_log/` is a manual step the user owns; help only when explicitly asked in a separate turn.
- Does **not** edit or delete existing entries in `top5_log/`.
