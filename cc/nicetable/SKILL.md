---
name: nicetable
description: Reformat markdown tables so every row's pipes line up. Pads each cell to its column's widest content and rebuilds the `---` separator to match. Operates on the user's IDE selection, a pasted block, or a file:line-range. Invoke as /nicetable [target].
user_invocable: true
---

# nicetable

Goal shape:

```
| Alias                    | Path                                                  |
|--------------------------|-------------------------------------------------------|
| `<webds_image_reg_text>` | `s3://nv-00-10206-vfm/webdataset_image_regular_text/` |
| `<webds_image_reg>`      | `s3://nv-00-10206-vfm/webdataset_image_regular/`      |
```

## Target

In order: explicit `path[:L1-L2]` arg → IDE selection → pasted `|...|` block in the message → ask. File/selection: edit in place. Pasted: reply with the reformatted block, no writes.

## Tool

A reusable Python implementation lives next to this file as `reformat.py`. **Prefer this over reimplementing the rules each time.** Usage:

- `python3 ~/.claude/skills/nicetable/reformat.py <path>` — reformat every table in the file in place.
- `python3 ~/.claude/skills/nicetable/reformat.py <path>:L1-L2 [<path2>:L3-L4 ...]` — only reformat tables overlapping that line range; multiple targets allowed.
- `python3 ~/.claude/skills/nicetable/reformat.py -` (or no args, with stdin) — read a pasted block on stdin, write the reformatted block to stdout.

The script implements the rules below. Use it for files. For pasted blocks where the reply must show the reformatted text, pipe through stdin mode.

## Behavior

For each contiguous run of `|...|` lines (sharing leading indent):

1. Split on `|` (not `\|`); drop the empty outer fields. **Strip all leading/trailing whitespace** per cell — markdown table cells don't carry padding as content, so this lets the script shrink an over-padded table back to its real width.
2. Detect the separator row (cells matching `:?-+:?`); the row before it is the header, the rows after are the body.
3. Re-render via `tabulate(body, headers=header, tablefmt='github', colalign=...)`. This auto-computes column widths from actual content and emits `|---|` plain-dash separators (matching the existing style in this codebase).
4. Preserve the table's leading indentation. Don't touch surrounding lines.

Don't change cell contents — only padding. Don't fix typos or content.

**Dependency:** requires `pip install tabulate` (already installed user-wide).

**Malformed input:** if the source line has the header and separator concatenated (e.g. one 400-char line containing both `| h1 | h2 |` and `|---|---|`), tabulate will read it as a single row with many cells and re-emit it that way. That correctly surfaces the corruption rather than papering over it. Fix the source markdown.
