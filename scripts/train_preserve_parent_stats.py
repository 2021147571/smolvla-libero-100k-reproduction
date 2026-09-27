"""Scoped training entrypoint; retain parent processor statistics without editing LeRobot."""
import json,sys
from pathlib import Path
import torch
from safetensors.torch import load_file
from lerobot.scripts import lerobot_train

PARENT=Path('/root/autodl-tmp/outputs/smolvla_libero_100k/checkpoints/100000/pretrained_model')
original=lerobot_train.make_pre_post_processors
def preserved(*args,**kwargs):
    assert Path(kwargs['pretrained_path']).resolve()==PARENT.resolve()
    kwargs.pop('dataset_stats',None)
    for key in ('preprocessor_overrides','postprocessor_overrides'):
        overrides={k:dict(v) for k,v in kwargs.get(key,{}).items()}
        for value in overrides.values(): value.pop('stats',None)
        kwargs[key]=overrides
    processors=original(*args,**kwargs)
    for pipeline,prefix in zip(processors,('policy_preprocessor','policy_postprocessor')):
        cfg=json.loads((PARENT/f'{prefix}.json').read_text())
        for step,entry in zip(pipeline.steps,cfg['steps'],strict=True):
            if entry['registry_name'] not in ('normalizer_processor','unnormalizer_processor'):continue
            expected=load_file(PARENT/entry['state_file']);actual=step.state_dict()
            assert set(expected)==set(actual),(prefix,'stat keys differ')
            assert all(torch.equal(expected[k].cpu(),actual[k].cpu()) for k in expected),(prefix,'stat values differ')
    print('PARENT_PROCESSOR_STATS_EXACTLY_PRESERVED',flush=True)
    return processors

if __name__=='__main__':
    torch.set_autocast_dtype('cuda',torch.bfloat16)
    lerobot_train.make_pre_post_processors=preserved
    if '--preflight-only' in sys.argv:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        cfg=SmolVLAConfig.from_pretrained(str(PARENT))
        fake=json.loads(Path('/root/autodl-tmp/datasets/local/smolvla_targeted_mix_v2_31d/meta/stats.json').read_text())
        preserved(policy_cfg=cfg,pretrained_path=str(PARENT),dataset_stats=fake,
                  preprocessor_overrides={'device_processor':{'device':'cuda'},'normalizer_processor':{'stats':fake}},
                  postprocessor_overrides={'unnormalizer_processor':{'stats':fake}})
    else:
        lerobot_train.main()
