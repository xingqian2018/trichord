#!/usr/bin/env bash
# Inspect a fixed set of ~/Project/* git repos and emit one pipe-delimited row per repo.
# Row format: <machine>|<repo>|<branch>|<unstaged yes/no>|<sync vs remote>
# Sync values: - (on spot), behind, lead, diverged, untracked (no upstream configured AND no origin/<branch> ref to fall back to).
# Missing repos (no .git) are skipped entirely — no row is emitted.
# Runs `git fetch --quiet` per repo first so the sync comparison is fresh, not stale.

set -u

MACHINE="${1:-$(hostname)}"
REPOS=(
  "$HOME/Project/imaginaire4"
  "$HOME/Project/imaginaire4_alt"
  "$HOME/Project/imaginaire4_sila"
  "$HOME/Project/bashrc"
  "$HOME/Project/trichord"
)

for dir in "${REPOS[@]}"; do
  [ -d "$dir/.git" ] || continue
  name=$(basename "$dir")
  branch=$(git -C "$dir" rev-parse --abbrev-ref HEAD 2>/dev/null)

  if [ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]; then
    unstaged=yes
  else
    unstaged=no
  fi

  # Refresh remote tracking refs (best-effort; ignore network/auth failures).
  git -C "$dir" fetch --quiet 2>/dev/null

  # Prefer the configured upstream. If the branch has none, fall back to
  # origin/<current-branch> when that ref exists — many repos have an origin
  # but never ran `git push -u`, and we still want a meaningful sync status.
  ref=""
  if git -C "$dir" rev-parse --verify --quiet '@{upstream}' >/dev/null 2>&1; then
    ref='@{upstream}'
  elif git -C "$dir" rev-parse --verify --quiet "refs/remotes/origin/${branch}" >/dev/null 2>&1; then
    ref="refs/remotes/origin/${branch}"
  fi

  if [ -z "$ref" ]; then
    sync="untracked"
  else
    counts=$(git -C "$dir" rev-list --left-right --count "HEAD...${ref}" 2>/dev/null)
    ahead=$(echo "$counts" | awk '{print $1}')
    behind=$(echo "$counts" | awk '{print $2}')
    if [ "$ahead" = "0" ] && [ "$behind" = "0" ]; then
      sync="-"
    elif [ "$ahead" != "0" ] && [ "$behind" = "0" ]; then
      sync="lead"
    elif [ "$ahead" = "0" ] && [ "$behind" != "0" ]; then
      sync="behind"
    else
      sync="diverged"
    fi
  fi

  echo "${MACHINE}|${name}|${branch}|${unstaged}|${sync}"
done
