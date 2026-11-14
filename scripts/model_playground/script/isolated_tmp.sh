#!/bin/bash
# Run a command with a private /tmp: a new mount namespace (unprivileged, via a user namespace) with a fresh tmpfs on /tmp.
# Every srun task on a node shares one Pyxis container, so without this they all share /tmp.
# Usage: bash script/isolated_tmp.sh python3 script/run_poster_generation_v1.py --output ...
exec unshare --user --map-root-user --mount bash -c 'mount -t tmpfs -o mode=1777 tmpfs /tmp && exec "$@"' isolated_tmp "$@"
