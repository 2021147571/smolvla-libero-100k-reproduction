import json
from pathlib import Path
from datetime import datetime, timezone

root = Path('/root/autodl-tmp')
groups = {}
for model in ('parent', 'teacher'):
    for h in (1, 10, 50):
        suites = {}
        for suite in ('spatial', 'object', 'goal'):
            path = root / f'outputs/paired_horizon_{model}_h{h}_{suite}_quick90/eval_info.json'
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            tasks = {str(t['task_id']): t['metrics']['successes'] for t in data['per_task']}
            assert len(tasks) == 10 and all(len(v) == 3 for v in tasks.values())
            suites[suite] = {'successes': sum(sum(v) for v in tasks.values()), 'episodes': 30, 'tasks': tasks}
        groups[f'{model}_h{h}'] = {'complete': len(suites) == 3, 'successes': sum(s['successes'] for s in suites.values()), 'suites': suites}
pairs = {}
for h in (1, 10, 50):
    p, t = groups[f'parent_h{h}'], groups[f'teacher_h{h}']
    if not (p['complete'] and t['complete']):
        continue
    rows = []
    for suite in ('spatial', 'object', 'goal'):
        for task in range(10):
            a, b = p['suites'][suite]['tasks'][str(task)], t['suites'][suite]['tasks'][str(task)]
            rows.append({'suite': suite, 'task_id': task, 'parent': sum(a), 'teacher': sum(b), 'teacher_only': sum(not x and y for x,y in zip(a,b)), 'parent_only': sum(x and not y for x,y in zip(a,b)), 'net': sum(b)-sum(a)})
    pairs[str(h)] = rows
report = {'timestamp': datetime.now(timezone.utc).isoformat(), 'groups': groups, 'matched_task_pairs': pairs}
(root / 'PAIRED_HORIZON_SUMMARY.json').write_text(json.dumps(report, indent=2))
lines = [f'{k}: complete={v["complete"]}, successes={v["successes"]}, suites=' + str({s:r['successes'] for s,r in v['suites'].items()}) for k,v in groups.items()]
for h, rows in pairs.items():
    lines.append(f'h{h}: teacher-only={sum(r["teacher_only"] for r in rows)}, parent-only={sum(r["parent_only"] for r in rows)}')
summary = '\n'.join(lines)
print(summary)
md = ['# Matched-horizon Quick-90 results', '', 'Development states 10–12, seed 1000, batch 1, AMP false, num_steps 10. Task IDs below are suite-local. No Short-300 has been launched.', '', '```', summary, '```', '', '## Per-task paired results', '', 'Each cell: parent successes / teacher successes (teacher minus parent), out of 3.', '', '| Suite | Task | h1 | h10 | h50 |', '|---|---:|---|---|---|']
for suite in ('spatial', 'object', 'goal'):
    for task in range(10):
        cells = []
        for h in (1, 10, 50):
            row = next((r for r in pairs.get(str(h), []) if r['suite'] == suite and r['task_id'] == task), None)
            cells.append(f'{row["parent"]} / {row["teacher"]} ({row["net"]:+d})' if row else 'pending')
        md.append(f'| {suite} | {task} | ' + ' | '.join(cells) + ' |')
md += ['', '## Interpretation', '', 'Teacher has no aggregate advantage at any matched horizon on this development set. Do not resume the abandoned fine-tuning route based on these results. Camera/preprocessing compatibility and article checkpoint identity remain unresolved. Suggested common formal protocol is horizon 10, based on development results and runtime, pending review; do not choose seeds or change formal parameters based on test scores. Server remains on by user request.']
(root / 'PAIRED_HORIZON_RESULTS.md').write_text('\n'.join(md) + '\n')
with (root / 'REPRODUCTION_LOG.md').open('a') as f:
    f.write('\n## Paired horizon progress ' + report['timestamp'] + '\n\n```\n' + summary + '\n```\nDevelopment states10-12 only; incomplete groups are not final results. No parameters changed.\n')
