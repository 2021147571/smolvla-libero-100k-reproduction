"""Frozen-parent camera pilot, optionally with official LVSM; no training."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import json, time, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import eval_support as core
import argparse
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser()
parser.add_argument('--medium',action='store_true')
parser.add_argument('--nvs',action='store_true')
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--run-tag',default='')
parser.add_argument('--parent',required=True,help='Local parent pretrained_model directory')
parser.add_argument('--output',required=True,help='New output directory; must not exist')
args=parser.parse_args()
core.BASE=Path(args.parent)
condition='medium_lvsm' if args.nvs else ('combined_medium' if args.medium else 'shift_y_10cm')
ROOT=Path(args.output)
if args.nvs:
    assert args.medium
if args.run_tag: ROOT=ROOT.with_name(ROOT.name+'_'+args.run_tag)
ROOT.mkdir(exist_ok=False)
def medium_pose(pos,quat,height):
    forward=-Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()[:,2]
    distance=(height-pos[2])/forward[2]
    assert distance>0
    center=pos+distance*forward
    rel=pos-center
    radius=np.linalg.norm(rel)
    theta=np.arctan2(rel[1],rel[0])+np.deg2rad(30)
    phi=np.arcsin(rel[2]/radius)+np.deg2rad(5)
    new_pos=center+(radius+.1)*np.array([np.cos(phi)*np.cos(theta),np.cos(phi)*np.sin(theta),np.sin(phi)])
    direction=center-new_pos
    direction/=np.linalg.norm(direction)
    right=np.cross(direction,[0,0,1]); right/=np.linalg.norm(right)
    up=np.cross(right,direction)
    new_quat=Rotation.from_matrix(np.column_stack([right,up,-direction])).as_quat()[[3,0,1,2]]
    assert np.isclose(np.linalg.norm(new_pos-center)-radius,.1)
    return new_pos,new_quat,center
def emit(event, **kw):
    row=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),event=event,**kw)
    with (ROOT/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

torch.set_num_threads(4)
policy=core.SmolVLAPolicy.from_pretrained(str(core.BASE)).to('cuda').eval()
policy.requires_grad_(False)
policy.config.n_action_steps=1
policy.config.num_steps=10
policy.config.use_amp=False
sha=core.digest(core.BASE/'model.safetensors')
pre,post=core.make_pre_post_processors(policy.config,pretrained_path=str(core.BASE),preprocessor_overrides={'device_processor':{'device':'cuda'}})
protocol=dict(tasks=[['spatial',0],['object',0],['goal',0]],states=list(range(30,35)),horizon=1,flow_steps=10,amp=False,batch=1,seed_rule='1000+state',shift_world_m=[0,.1,0],camera='agentview',parent_sha256=sha,scope='custom small pilot, not paper reproduction; no training')
if args.medium:
    protocol.pop('shift_world_m')
    protocol.update(condition=condition,delta_radius=.1,delta_theta=30,delta_phi=5,
        workspace_height={'spatial':.912,'object':0.,'goal':.912},
        geometry_source='https://github.com/heo0224/AnyCamVLA/tree/main/openpi/examples/libero',
        reference='canonical camera central ray intersecting official task robot_base_z plane; camera looks at reference')
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
adapter=None
if args.nvs:
    protocol.update(nvs='official LVSM 20k checkpoint; BF16',native_image_size=360,nvs_image_size=256,
        output_resize='bilinear 256 to 360',input_flip='vertical native to LVSM; inverse on output',
        unchanged_camera='wrist',baseline='results/original_and_medium.json')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    from anycam_adapter import Adapter
    adapter=Adapter(ROOT)
results=[]
started=time.monotonic()
try:
    for suite,task in protocol['tasks']:
        cfg=core.LiberoEnv(task='libero_'+suite,task_ids=[task],init_state_offset=30)
        env=next(iter(next(iter(core.make_env(cfg,n_envs=1,use_async_envs=False).values())).values()))
        epre,epost=core.make_env_pre_post_processors(cfg,policy.config)
        u=env.envs[0].unwrapped
        original_reset=u.reset
        mode={'shift':False}
        def camera_reset(*reset_args,**kwargs):
            obs,info=original_reset(*reset_args,**kwargs)
            robot_env=u._env.env
            sim=robot_env.sim
            cid=sim.model.camera_name2id('agentview')
            # Restore original pose even when the simulator uses a soft reset.
            if not hasattr(u,'_pair_original_pose'):
                u._pair_original_pose=(sim.model.cam_pos.copy(),sim.model.cam_quat.copy())
            poses,quats=u._pair_original_pose
            sim.model.cam_pos[:]=poses
            sim.model.cam_quat[:]=quats
            sim.forward()
            before=sim.get_state().flatten().copy()
            assert int(sim.model.cam_bodyid[cid])==0, 'Shift assumes world-fixed agent camera'
            if mode['shift']:
                if args.medium:
                    p,q,center=medium_pose(poses[cid],quats[cid],protocol['workspace_height'][suite])
                    sim.model.cam_pos[cid]=p
                    sim.model.cam_quat[cid]=q
                else: sim.model.cam_pos[cid]+=np.array([0,.1,0])
            sim.forward()
            assert np.array_equal(before,sim.get_state().flatten()), 'Camera edit changed physics state'
            others=np.arange(len(poses))!=cid
            assert np.array_equal(sim.model.cam_quat[others],quats[others])
            assert np.array_equal(sim.model.cam_pos[others],poses[others])
            raw=robot_env._get_observations(force_update=True)
            obs=u._format_raw_obs(raw)
            u._pair_state=before
            u._pair_pose=dict(position=sim.model.cam_pos[cid].tolist(),quaternion=sim.model.cam_quat[cid].tolist(),fovy=float(sim.model.cam_fovy[cid]))
            return obs,info
        u.reset=camera_reset
        try:
            for state in protocol['states']:
                paired_state=None
                for label in ([condition] if args.nvs else ['original',condition]):
                    mode['shift']=label!='original'
                    seed=1000+state
                    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                    u.init_state_id=state
                    policy.reset()
                    obs,_=env.reset(seed=[seed],options={core.NEW_ROLLOUT_OPTION:True})
                    assert u.init_state_id==state+1
                    if paired_state is None: paired_state=u._pair_state.copy()
                    else: assert np.array_equal(paired_state,u._pair_state), 'Unmatched paired initial physics state'
                    pixels=obs['pixels']
                    if state==30:
                        for name,arr in pixels.items():
                            Image.fromarray(np.asarray(arr)[0]).save(ROOT/f'{suite}_{label}_{name}.png')
                    emit('episode_start',suite=suite,task=task,state=state,condition=label,camera=u._pair_pose,task_description=env.call('task_description')[0])
                    t=time.monotonic(); won=False
                    for step in range(env.call('_max_episode_steps')[0]):
                        if adapter:
                            obs=adapter.apply(obs,u,suite,state)
                            if args.smoke:
                                emit('smoke_complete',suite=suite,state=state)
                                raise SystemExit(0)
                        with torch.inference_mode():
                            batch=core.preprocess_observation(obs)
                            batch['task']=list(env.call('task_description'))
                            action=policy.select_action(pre(epre(batch)))
                            action=epost({'action':post(action)})['action']
                            obs,reward,terminated,truncated,info=env.step(action.cpu().numpy())
                        won=won or core.success(info)
                        if won or bool(terminated[0]) or bool(truncated[0]): break
                    row=dict(suite=suite,task=task,state=state,seed=seed,condition=label,success=won,steps=step+1,seconds=time.monotonic()-t)
                    results.append(row)
                    (ROOT/'results.json').write_text(json.dumps(results,indent=2))
                    emit('episode_complete',**row)
        finally: env.close()
    assert core.digest(core.BASE/'model.safetensors')==sha
    totals={label:sum(r['success'] for r in results if r['condition']==label) for label in ([condition] if args.nvs else ['original',condition])}
    emit('complete',totals=totals,episodes=len(results),seconds=time.monotonic()-started,parent_unchanged=True)
except Exception as exc:
    emit('failed',error=repr(exc))
    raise
