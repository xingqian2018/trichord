---
name: cheatsheet_run
description: Run a named cheatsheet. Each cheatsheet is a sibling .md file in this skill's folder describing a specific recipe or workflow. Invoke as /cheatsheet_run <name>.
user_invocable: true
---

# cheatsheet_run

Each sibling `.md` file in this folder (everything except `SKILL.md`) is one cheatsheet. The filename (without `.md`) is the cheatsheet's name.

## How to handle an invocation

1. **Parse the argument.** The user passes the cheatsheet name as the argument (e.g. `/cheatsheet_run foo` → `foo.md`).
2. **If no argument was given**, list the available cheatsheets in this folder and ask the user which one to run. Do not guess.
3. **If the argument does not match any file**, list the available cheatsheets and ask the user to pick one. Do not fuzzy-match silently.
4. **Read the matching `<name>.md`** from this skill's folder.
5. **Follow the instructions in that file exactly.** The cheatsheet is authoritative — treat it the way you would treat direct user instructions for this task. Do not add steps it does not specify, and do not skip steps it does specify.
6. If the cheatsheet itself points at other files (sibling cheatsheets, external paths), read those as instructed.

## Listing cheatsheets

Available cheatsheets = every `*.md` file in `~/.claude/skills/cheatsheet_run/` other than `SKILL.md`. Use `ls` or `Bash` to enumerate them when needed; do not rely on memory.