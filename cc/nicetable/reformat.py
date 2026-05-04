#!/usr/bin/env python3
"""Reformat markdown tables so every row's pipes line up.

Usage:
    reformat.py <path>[:L1-L2] [<path2>[:L3-L4] ...]   # edit file(s) in place
    reformat.py -                                      # stdin -> stdout

Implementation: parses each contiguous run of `|...|` lines (sharing leading
indent) and re-renders it through `tabulate` with `tablefmt="github"`. The
github format produces plain markdown (`|---|`) with auto-computed column
widths. Cell contents are stripped of leading/trailing whitespace — markdown
table cells don't carry padding as content, so this lets us shrink an
over-padded table back to its real width.
"""

import re
import sys

from tabulate import tabulate

LINE_RE = re.compile(r'^(\s*)(\|.*\|)\s*$')
SEP_CELL_RE = re.compile(r'^:?-+:?$')


def split_cells(content):
    cells = []
    cur = ''
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == '\\' and i + 1 < len(content):
            cur += content[i:i + 2]
            i += 2
            continue
        if ch == '|':
            cells.append(cur)
            cur = ''
        else:
            cur += ch
        i += 1
    cells.append(cur)
    return [c.strip() for c in cells[1:-1]]


def is_separator(cells):
    return bool(cells) and all(SEP_CELL_RE.match(c) for c in cells)


def alignment_for(cell):
    left = cell.startswith(':')
    right = cell.endswith(':')
    if left and right:
        return 'center'
    if right:
        return 'right'
    return 'left'


def format_block(rows, indent):
    ncols = max(len(r) for r in rows)
    rows = [r + [''] * (ncols - len(r)) for r in rows]

    sep_idx = next((i for i, r in enumerate(rows) if is_separator(r)), None)
    if sep_idx is None:
        header = rows[0]
        body = rows[1:]
        aligns = ['left'] * ncols
    else:
        header = rows[sep_idx - 1] if sep_idx > 0 else [''] * ncols
        body = rows[sep_idx + 1:]
        aligns = [alignment_for(c) for c in rows[sep_idx]]

    out = tabulate(body, headers=header, tablefmt='github', colalign=aligns)
    return [indent + line for line in out.splitlines()]


def process(text):
    lines = text.split('\n')
    out = []
    i = 0
    n = len(lines)
    while i < n:
        m = LINE_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group(1)
        block = []
        while i < n:
            mm = LINE_RE.match(lines[i])
            if not mm or mm.group(1) != indent:
                break
            block.append(split_cells(mm.group(2)))
            i += 1
        if len(block) >= 2:
            out.extend(format_block(block, indent))
        else:
            for r in block:
                out.append(indent + '| ' + ' | '.join(r) + ' |')
    return '\n'.join(out)


def parse_target(arg):
    m = re.match(r'^(.*?)(?::(\d+)-(\d+))?$', arg)
    path = m.group(1)
    if m.group(2) is None:
        return path, None, None
    return path, int(m.group(2)), int(m.group(3))


def process_file(path, l1=None, l2=None):
    with open(path, 'r') as f:
        txt = f.read()
    if l1 is None:
        new = process(txt)
    else:
        lines = txt.split('\n')
        l1 = max(1, l1)
        l2 = min(len(lines), l2)
        head = lines[:l1 - 1]
        mid = lines[l1 - 1:l2]
        tail = lines[l2:]
        mid_new = process('\n'.join(mid)).split('\n')
        new = '\n'.join(head + mid_new + tail)
    if new != txt:
        with open(path, 'w') as f:
            f.write(new)
        return True
    return False


def main():
    args = sys.argv[1:]
    if not args or args == ['-']:
        sys.stdout.write(process(sys.stdin.read()))
        return
    for arg in args:
        path, l1, l2 = parse_target(arg)
        changed = process_file(path, l1, l2)
        print(f'{"updated" if changed else "unchanged"}: {arg}')


if __name__ == '__main__':
    main()
