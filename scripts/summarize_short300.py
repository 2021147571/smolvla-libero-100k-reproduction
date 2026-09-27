import json
from pathlib import Path

root = Path("/root/autodl-tmp")
suites = ("spatial", "object", "goal")
summary = {"checkpoint": "100000", "seed": 1000, "batch_size": 1, "suites": {}}
lines = [
    "# SmolVLA 100k Short-300 评估结果",
    "",
    "- 权重：微调前 100k checkpoint",
    "- 套件：LIBERO Spatial / Object / Goal",
    "- 每个套件 10 个任务，每任务 10 条轨迹",
    "- seed：1000；batch size：1；三个套件独立并行运行",
    "",
    "| Suite | 成功数 | 轨迹数 | 成功率 |",
    "|---|---:|---:|---:|",
]
total_success = 0
total_eps = 0
task_rows = []
for suite in suites:
    path = root / f"outputs/short300_100k_{suite}_seed1000_b1/eval_info.json"
    data = json.loads(path.read_text())
    tasks = []
    suite_success = 0
    for entry in data["per_task"]:
        successes = [bool(x) for x in entry["metrics"]["successes"]]
        count = sum(successes)
        suite_success += count
        tasks.append({"task_id": entry["task_id"], "successes": count, "episodes": len(successes), "outcomes": successes})
        task_rows.append(f"| {suite} | {entry['task_id']} | {count}/10 | {count * 10}% |")
    episodes = sum(t["episodes"] for t in tasks)
    summary["suites"][suite] = {"successes": suite_success, "episodes": episodes, "success_rate": suite_success / episodes, "tasks": tasks}
    total_success += suite_success
    total_eps += episodes
    lines.append(f"| {suite.title()} | {suite_success} | {episodes} | {suite_success / episodes:.0%} |")

summary["overall"] = {"successes": total_success, "episodes": total_eps, "success_rate": total_success / total_eps}
lines += [
    f"| **Overall** | **{total_success}** | **{total_eps}** | **{total_success / total_eps:.1%}** |",
    "",
    "## 各任务成绩",
    "",
    "| Suite | Task ID | 成功数 | 成功率 |",
    "|---|---:|---:|---:|",
    *task_rows,
    "",
    "## 说明",
    "",
    "三个 suite 为缩短墙钟时间而分别启动独立进程；每个 suite 内仍严格使用 batch_size=1。SmolVLA 推理包含随机 flow-matching 采样，因此改变 batch 或进程边界可能改变随机动作序列。本结果的可复现口径为上述 seed、batch 和分组方式；与文章比较前仍应确认其随机种子及运行脚本。",
]
(root / "outputs/SHORT300_BASELINE_100K.md").write_text("\n".join(lines) + "\n")
(root / "outputs/short300_baseline_100k_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

log = root / "REPRODUCTION_LOG.md"
with log.open("a", encoding="utf-8") as f:
    f.write("\n## 2026-09-23 Short-300 评估结果\n\n")
    f.write(f"40. 微调前 100k checkpoint 的 Short-300 完成：Spatial {summary['suites']['spatial']['successes']}/100（{summary['suites']['spatial']['success_rate']:.0%}）、Object {summary['suites']['object']['successes']}/100（{summary['suites']['object']['success_rate']:.0%}）、Goal {summary['suites']['goal']['successes']}/100（{summary['suites']['goal']['success_rate']:.0%}），总计 {total_success}/{total_eps}（{total_success/total_eps:.1%}）。\n")
    f.write("41. 为缩短墙钟时间，三个 suite 使用独立进程并行，每个 suite 内保持 batch_size=1、seed=1000、每任务 10 条固定初始状态；完整逐任务结果见 `/root/autodl-tmp/outputs/SHORT300_BASELINE_100K.md`。因 SmolVLA 动作采样具有随机性，与文章比较前仍需核实其 seed 和进程/批处理口径。\n")
