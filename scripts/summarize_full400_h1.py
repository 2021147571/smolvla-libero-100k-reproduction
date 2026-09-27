import json, hashlib, shutil
from pathlib import Path
r=Path('/root/autodl-tmp'); dest=r/'reports/full400_h1'
dest.mkdir(parents=True,exist_ok=True)
results={}; manifest={}
lines=['# Full400 horizon1 paired evaluation','', 'Both: execution horizon1, flow-matching steps10, seed1000, AMPfalse, batch1, init states0-9. Each model keeps its native processors. Teacher revision31d453f7.','', '| Suite | Parent | Teacher | Teacher minus parent |','|---|---:|---:|---:|']
for suite in ['spatial','object','goal','10']:
    models={}
    for model in ['parent','teacher31d']:
        src=r/f'outputs/full400_h1_{model}_{suite}_seed1000/eval_info.json'
        data=json.loads(src.read_text()); tasks=data['per_task']
        assert len(tasks)==10 and sorted(t['task_id'] for t in tasks)==list(range(10))
        models[model]={t['task_id']:t['metrics']['successes'] for t in tasks}
        assert all(len(v)==10 and all(type(x) is bool for x in v) for v in models[model].values())
        target=dest/f'{model}_{suite}_eval_info.json'; shutil.copy2(src,target)
        manifest[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    p,t=models['parent'],models['teacher31d']
    rows=[dict(task_id=i,parent=sum(p[i]),teacher=sum(t[i]),parent_outcomes=p[i],teacher_outcomes=t[i],teacher_only=sum(b and not a for a,b in zip(p[i],t[i])),parent_only=sum(a and not b for a,b in zip(p[i],t[i]))) for i in range(10)]
    a=sum(x['parent'] for x in rows); b=sum(x['teacher'] for x in rows)
    results[suite]=dict(parent=a,teacher=b,tasks=rows)
    lines.append(f'| {suite} | {a}/100 | {b}/100 | {b-a:+d} |')
a=sum(v['parent'] for v in results.values()); b=sum(v['teacher'] for v in results.values())
lines += [f'| Total | {a}/400 ({a/4:.2f}%) | {b}/400 ({b/4:.2f}%) | {b-a:+d} |','','## Task-level results','','| Suite/task | Parent /10 | Teacher /10 | Teacher-only | Parent-only |','|---|---:|---:|---:|---:|']
for s,d in results.items():
    for t in d['tasks']: lines.append(f"| {s}/{t['task_id']} | {t['parent']} | {t['teacher']} | {t['teacher_only']} | {t['parent_only']} |")
lines+=['','Episode-index pairing uses corresponding initial states, but sequential stochastic RNG may diverge. These are observed results under this implementation, not proof of exact paper reproduction. Test states have already been inspected, not an untouched future confirmation set.','','Completed22:41:43 +08:00 on2026-09-27. No finetuning or correction collection authorized now. Keep cloud and local computer ON; wait for user.']
(dest/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
(dest/'COMPARISON.json').write_text(json.dumps(results,indent=2))
note=f'\n\n## Full400 h1 complete2026-09-27 22:41:43 +08:00\nValidated40tasks x10Boolean outcomes per model. Parent {a}/400; teacher31d {b}/400. Report reports/full400_h1/COMPARISON.md includes suite/task and init-index differences. No finetuning or new experiments; keep server and computer ON and wait for user.\n'
for name in ['TEACHER_REAUDIT.md','REPRODUCTION_LOG.md']:
    with (r/name).open('a') as f: f.write(note)
    shutil.copy2(r/name,dest/name)
for f in dest.iterdir():
    if f.name!='SHA256.json': manifest[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(dest/'SHA256.json').write_text(json.dumps(manifest,indent=2))
print('\n'.join(lines[:11]))
