---
name: ssh_run
description: 在远端集群（awscode / gcpcode / gb300）上通过 SSH 执行命令或提交 slaunch / Slurm 任务。本地写启动脚本 → scp（-S ~/Software/cluster_run）到远端 ~/tmp → ~/Software/cluster_run 执行，提交后报告 Slurm job id 和远端日志位置。用于任何"在集群上跑 X / 提交任务 / 查任务状态"的请求。
user_invocable: true
---

# 第一步：先读规则 ⚠️

## 不要直接写 `ssh` / `scp`，用 `~/Software/cluster_run`

- 连远端一律用 `~/Software/cluster_run`。它是 ssh 客户端的一个副本，参数和 ssh 完全一样：`~/Software/cluster_run <host> '<远端命令>'`。
- 拷文件用 `/usr/bin/scp -S ~/Software/cluster_run <本地文件> <host>:<远端路径>`（`-S` 让 scp 走 cluster_run）。
- 直接写 `ssh` / `/usr/bin/ssh` 会被权限设置拒绝；`scp` 也统一加 `-S ~/Software/cluster_run`。

## `scancel` 规则（任何 Slurm 任务都适用）

- **未经用户对具体 job id 明确回答"是"，绝不执行 `scancel`。** 以下情况同样适用：
  - "重新提交 / 重跑 / 修完再提交"——即使用户暗示之前的任务有问题，也要先问。问法：*"这需要 `scancel <jobids>`，确认吗？"*
  - 取消自己在本对话里刚提交的任务——也要问。
  - 查重时发现的重复任务——也要问。
  - "明显错了"的任务（集群错、凭证错、节点数错）——也要问。
- **不要默认用户想杀掉旧任务。** 新旧任务可能需要并存。确认时贴出准确的 job id。
- **原因：** 杀错任务会浪费数小时到数天的算力，而且无法恢复。"重跑"≠"先 scancel 再重跑"，除非用户明确这么说。

## 主机

| 主机 | 说明 |
|---|---|
| `awscode` | AWS 集群登录节点 |
| `gcpcode` | GCP 集群登录节点 |
| `gb300` | GB300 集群登录节点 |

按用户说法选："aws" → `awscode`，"gcp" → `gcpcode`，"gb300" → `gb300`。不明确时先问。

## 提交前先给用户看命令

完整的启动命令（多行展开）先贴给用户，用户说"提交 / 执行 / 跑"之后再提交。用户已经明确说过"直接跑"的除外。

## 提交前必须查重

```bash
~/Software/cluster_run <host> 'squeue -u $USER -o "%i %j %T %R"' | grep <任务名>
```

- 按任务名和关键参数（输出路径、`--signature` 等）匹配。
- 一批多个变体时，每个都要查，不能只查一个。
- 发现重复就停下来问：跳过 / 取消后重跑 / 照样提交。

## 远端登录 shell 是 csh

- 在 ssh 引号里写 `2>/dev/null`、`VAR=x cmd`、`export`、复杂的双引号嵌套，都会报错（例如 `Ambiguous output redirect`、`Unmatched "`）。
- 管道、grep、重定向尽量放在**本地**一侧：`~/Software/cluster_run gcpcode 'cmd' | grep xxx`。
- 一定要在远端用 bash 语法时，写成脚本再 `bash ~/tmp/<name>.sh`。

# 第二步：怎么执行

## 简单的单行命令

```bash
~/Software/cluster_run <host> '<远端命令>'
```

- 远端命令用**单引号**包起来，避免本地 shell 展开 `$` 和反引号。
- 远端命令本身含单引号时，用 `'"'"'` 转义。
- 需要特定目录时串一个 `cd`：`cd <dir> && <cmd>`。

## slaunch / Slurm / 其他多行命令（标准流程）

### 1. 本地写启动脚本

用 Write 工具写到本地 `~/tmp/sshrun/<任务名>.sh`，固定格式：

```bash
#!/bin/bash
shopt -s expand_aliases
source ~/Project/bashrc/bashrc_my.sh

cd /home/xingqianx/Project/imaginaire4_sila

CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 16x4 <任务名> \
    pipelines/image/text_rendering/<脚本>.py \
    --参数1 ... \
    --参数2 ...
```

- 前两行不能省：`slaunch` 是 bashrc 里的 alias，非交互 bash 默认不展开 alias。
- 写完用 `bash -n ~/tmp/sshrun/<任务名>.sh` 检查语法。
- `slaunch` 的第二个参数必须是 `.py` 路径，不能是 `bash -c "..."`（slaunch 会把它包进 torchrun）。
- 节点规格写法：`cpu NxM`（N 个节点、每节点 M 个进程）。gcp 的 cpu 分区排队很深时可改用 `ulong`（每节点固定 4 个进程，一次性单进程脚本要加 rank 判断）。

### 2. 查重 → 拷贝 → 提交

```bash
~/Software/cluster_run <host> 'squeue -u $USER -o "%i %j %T %R"' | grep <任务名>
~/Software/cluster_run <host> 'mkdir -p ~/tmp'
/usr/bin/scp -S ~/Software/cluster_run ~/tmp/sshrun/<任务名>.sh <host>:~/tmp/<任务名>.sh
~/Software/cluster_run <host> 'bash ~/tmp/<任务名>.sh' 2>&1 | grep -o "Submitted batch job [0-9]*"
```

- 远端 `~/tmp/` 是持久目录，用完不删，方便事后查看。
- 不要用 `/tmp/` 当临时目录。

### 3. slaunch 在集群上做了什么（以 gcpcode 为例）

`slaunch` → `~/Project/bashrc/sbatch_launch/main.sh` → `sbatch_launch_gcp_cpu.sh`：

- `srun` 起 pyxis 容器，挂载 `/lustre/fsw/portfolios/cosmos` 和 `$HOME`。
- 容器工作目录 = `CONTAINER_WORKDIR`，并设置 `PYTHONPATH=.`（所以仓库里的 `pipelines.*` 可以直接 import）。
- 用 `.venv/bin/python -m torch.distributed.run`（c10d rendezvous）跑目标脚本。
- 默认排除坏节点 `cpu-large-025`；临时排除别的节点用 `SBATCH_EXCLUDE=<node>`。
- 日志写在远端 `~/log/slurm/<任务名>.<jobid>.e` 和 `.o`。

### 4. 代码从哪来

- **默认用远端已 checkout 的仓库**（`~/Project/imaginaire4_sila` 等）。代码同步（commit / push / pull、切分支、装环境）由用户负责，不要自己动集群上的仓库。
- **绝不把文件 scp 进集群上的仓库目录。**
- **新代码还没同步时：** 把脚本放到远端 `~/tmp/<名字>/` 的**独立子目录**，启动脚本里传绝对路径：

  ```bash
  slaunch cpu 16x4 <任务名> \
      /home/xingqianx/tmp/<名字>/<脚本>.py \
      --参数 ...
  ```

  因为容器挂载了 `$HOME`、且 `PYTHONPATH=.` 指向仓库，脚本照样能 import 仓库里的依赖。用独立子目录是为了避免 `~/tmp` 下其他 `.py` 文件抢先被 import。

### Heredoc 禁令（绝对）

**任何地方都不要用 heredoc**：`<<EOF`、`<<-EOF`、`<<'EOF'`、`<<"EOF"`、自定义结束符都不行。

引号复杂时一律走"本地写文件 → scp → ssh 执行"：

- 用 Write 工具写本地 `~/tmp/sshrun/<name>.sh`
- `/usr/bin/scp -S ~/Software/cluster_run ~/tmp/sshrun/<name>.sh <host>:~/tmp/<name>.sh`
- `~/Software/cluster_run <host> 'bash ~/tmp/<name>.sh'`

**原因：** `~/Software/cluster_run host "cat > foo.sh <<'EOF' ... EOF; bash foo.sh"` 会同时出两个问题：反斜杠换行在 heredoc 收集前被远端 shell 吃掉导致行合并；行合并后 `EOF` 认不出来，`cat` 把后面的 `bash foo.sh` 也写进脚本，脚本递归调用自己，无限循环。

---

## 与其他 skill 组合

用户在提到 `ssh_run` 的同时提到另一个 skill（例如"ssh_run meow on awscode"）时：

- 读 `cc/<skill_name>/SKILL.md` 拿到准确的 shell 命令。
- 用 `~/Software/cluster_run <host> '<cmd>'` 在远端执行，**不要**在本地调用那个 skill。
- Slurm job id 报告规则照常适用。
- 如果那个 skill 的步骤很难翻译成一条远端命令，说明情况并询问。

反例：用户只调用另一个 skill、没提 `ssh_run`，就在本地正常执行。

---

# 第三步：报告

ssh 返回后：

- **一句话说明已提交：** `Launched on <host>: <简短描述>`。
- **不要转述远端 stdout**，只提取 Slurm job id。
- **拿到 job id**（`Submitted batch job <N>` 或单独的数字）时，按下面格式报告：

```
Launched on <host>: <任务名或命令>
Slurm job id: <jobid>
Remote log:   ~/log/slurm/<任务名>.<jobid>.e  (stderr)
              ~/log/slurm/<任务名>.<jobid>.o  (stdout)
To tail:      ~/Software/cluster_run <host> 'tail -f ~/log/slurm/<任务名>.<jobid>.e'
```

这个数字是 **Slurm job id**，不是 `slaunch` / `sbatch` 包装进程的 PID，统一标注为 "Slurm job id"。日志文件在**远端**。

- **ssh 返回非零：** 报告退出码和 stderr 最后几行，不要自动重试。

## 查状态（用户问"跑完了吗 / 怎么样"时）

不挂持续监控，用户问的时候查一次：

```bash
~/Software/cluster_run <host> 'sacct -j <jobid> -X -o JobID,State,Start,Elapsed,ExitCode'
~/Software/cluster_run <host> 'grep -v "it/s]" ~/log/slurm/<任务名>.<jobid>.e' | grep -E "Progress|SUCCESS|Task failed|Traceback|Error" | tail -10
```

- `.e` 日志里的 Progress / SUCCESS 行最能说明进度。
- 任务结束时常见的 `RendezvousConnectionError` / `TCPStore` 堆栈多半是退出阶段的噪音，先确认 `sacct` 状态和最后的 SUCCESS 行，再判断是否真失败。
- 结果文件（例如 S3 JSON）要实际下载下来核对，不只看日志。

## 本 skill 不做的事

- 不改本地或远端的仓库代码。
- 不主动轮询任务状态；用户要状态时用 `/checkrun` 或直接问。
- 不装、不改 crontab，除非用户写清楚了具体时间表和命令；说法模糊时先问。
- `awscode` / `gcpcode` 不明确时不替用户选，先问。
