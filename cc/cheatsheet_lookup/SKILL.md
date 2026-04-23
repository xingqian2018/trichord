---
name: cheatsheet_lookup
description: Look up a named cheatsheet and show its contents. Each cheatsheet is a sibling .md file in this skill's folder describing a specific recipe or workflow. Invoke as /cheatsheet_lookup <name>. This skill only reads and displays — it does not run anything.
user_invocable: true
---

# cheatsheet_lookup

Sibling `.md` files (except `SKILL.md`) are cheatsheets. Filename without `.md` is the name.

**Lookup-only.** Display contents; never execute, launch, or modify anything.

## Steps

1. Parse the argument as `<name>` → `<name>.md`.
2. If no argument or no match, list available cheatsheets (`ls` the folder) and ask the user to pick. No guessing, no fuzzy-matching.
3. Read and show the matching file.
4. **Keep the template's indented multi-line form** (backslash continuations, `VAR=... \` prefixes, per-line args). Substitute placeholders the user gave you; do not flatten. Only produce a one-liner if the user explicitly asks.
5. **Resolve credentials locally** (env var, `credentials/*.json`, `~/.aws/credentials`, `~/.netrc`, etc.) and inline them into the command. If unresolvable, leave the `<placeholder>` and say where to set it.
6. Leave any unresolved placeholders as `<angle-bracket>` and call them out. Never invent values.
