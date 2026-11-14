#!/usr/bin/env bash
# "snapshot" on oci-jhb-slurm-2 = the two tables the user reads (user 2026-11-08 16:2x: "我们基本上就是看这两个表格"):
#   1. 节点使用      cluster_usage.sh  (normal / p0 / p0_tc / p0_xingqian / p0_runlong / p0_guard / p1 ... / reserved
#                    + the tc | xingqian 实验 | Runlong | ... line and the 其他人 ranking)
#   2. tc_full 分片  tc_shard_report.sh (0-899 / 900-1799 / 1800-2699 / 2700-3599: 完成 / 正在执行 / 执行跳过 / 排队 / 进度)
# plus ONE line under table 2 from tc_liveness.py (are the RUNNING tc tasks producing; 卡住 split by array).
# The three run in parallel. Table 2 reuses a cache scan younger than 20 min, else scans on a cpu-interactive node (~4 min).
# Read-only; writes only to ~/tmp/snapshot (runtime files stay out of this repo). Usage: snapshot.sh [--detail]   (--detail is passed to tc_shard_report)
set -uo pipefail
# The three helpers stay in the infra skill (other sessions keep editing cluster_usage.sh there; one copy only).
HERE=${SNAPSHOT_TOOLS:-$HOME/.claude/skills/oci-jhb-gb300-infra/scripts}
for f in cluster_usage.sh tc_shard_report.sh tc_shard_report.py tc_liveness.py; do
  [ -f "$HERE/$f" ] || { echo "snapshot: missing $HERE/$f (oci-jhb-slurm-2 login only)" >&2; exit 1; }
done
OUT=$HOME/tmp/snapshot
mkdir -p "$OUT"
T=$(date +%m%d_%H%M%S)
U=$OUT/usage_$T.txt S=$OUT/shard_$T.txt L=$OUT/live_$T.txt

bash "$HERE/tc_shard_report.sh" "$@" > "$S" 2>&1 &   pS=$!
bash "$HERE/cluster_usage.sh"         > "$U" 2>&1 &   pU=$!
(cd /tmp && python3 -I "$HERE/tc_liveness.py" --list) > "$L" 2>&1 &   pL=$!
wait $pU; rU=$?; wait $pL; rL=$?; wait $pS; rS=$?

echo "### 节点使用"
echo
[ $rU -eq 0 ] && cat "$U" || { echo "(cluster_usage.sh failed rc=$rU)"; tail -5 "$U"; }
echo
echo "### tc_full 分片"
echo
[ $rS -eq 0 ] && sed -n '/^tc_full 分片/,$p' "$S" || { echo "(tc_shard_report.sh failed rc=$rS)"; tail -5 "$S"; }
echo
if [ $rL -eq 0 ]; then
  python3 -I - "$L" <<'EOF'
import re, sys, collections
txt = open(sys.argv[1]).read()
c = dict(re.findall(r"^\| (产出中|加载中|慢|待确认|卡住) \| (\d+) \|", txt, re.M))
tot = re.search(r"^\| \*\*合计\*\* \| \*\*(\d+)\*\*", txt, re.M)
med = re.search(r"中位 ([\d,]+)", txt)
when = re.search(r"（(\d\d-\d\d \d\d:\d\d)", txt)
stuck = collections.Counter(m.split("_")[0] for m in re.findall(r"^\s+卡住 (\S+)", txt, re.M))
line = (f"吞吐（{when.group(1) if when else '?'}，在跑 {tot.group(1) if tot else '?'}）：产出中 {c.get('产出中', 0)} / 加载中 {c.get('加载中', 0)}"
        f" / 慢 {c.get('慢', 0)} / 待确认 {c.get('待确认', 0)} / 卡住 {c.get('卡住', 0)}；产出中位 {med.group(1) if med else '-'} tok/s")
if stuck:
    line += "；卡住按 array：" + ", ".join(f"{a} {n}" for a, n in stuck.most_common())
print(line)
EOF
else
  echo "(tc_liveness.py failed rc=$rL)"; tail -3 "$L"
fi
echo
echo "（原始输出：$U  $S  $L）"
find "$OUT" -name '*_*.txt' -mtime +2 -delete 2>/dev/null || true
