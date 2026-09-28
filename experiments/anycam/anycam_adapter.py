"""Official LVSM inference adapter for native LeRobot LIBERO image convention."""
import sys, os, time, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from easydict import EasyDict
SRC=Path(os.environ['ANYCAM_SOURCE']).resolve()
sys.path.insert(0,str(SRC/'LVSM'))
sys.path.insert(0,str(SRC/'openpi/examples/libero'))
from nvs import LVSMWrapper
from libero_utils import extract_camera_parameters
from model.LVSM_scene_decoder_only import Images2LatentScene

class Adapter:
    def __init__(self, root):
        self.root=root
        cfg=EasyDict(OmegaConf.to_container(OmegaConf.load(SRC/'LVSM/configs/inference_LVSM_LIBERO-Plus_custom.yaml'),resolve=True))
        # Loss-only modules never participate in has_target_image=False inference.
        cfg.training.perceptual_loss_weight=0.
        cfg.training.lpips_loss_weight=0.
        ckpt=SRC/'LVSM/ckpt/LIBERO-Plus_custom/ckpt_0000000000020000.pt'
        model=Images2LatentScene(cfg)
        weights=torch.load(ckpt,map_location='cpu',weights_only=True)['model']
        result=model.load_state_dict(weights,strict=False)
        assert not result.missing_keys,result.missing_keys
        assert all(k.startswith('loss_computer.') for k in result.unexpected_keys),result.unexpected_keys
        self.wrapper=LVSMWrapper.__new__(LVSMWrapper)
        self.wrapper.config=cfg
        self.wrapper.device='cuda'
        self.wrapper.amp_dtype_mapping={'bf16':torch.bfloat16}
        model.to('cuda')
        model.eval()  # Upstream train() override does not return self.
        model.requires_grad_(False)
        self.wrapper.model=model
        self.names=['agentview','robot0_eye_in_hand']
        self.saved=set()
        print('LVSM_STRICT_INFERENCE_WEIGHTS_OK',flush=True)

    def apply(self,obs,u,suite,state):
        env=u._env.env
        sim=env.sim
        mapping=u.camera_name_mapping
        # LeRobot leaves the native raw camera array unchanged; official AnyCam
        # inputs are raw[::-1] (its 180-degree flip followed by horizontal flip).
        keys=[mapping[n+'_image'] if n+'_image' in mapping else mapping[n] for n in self.names]
        raw=[np.asarray(obs['pixels'][k])[0] for k in keys]
        h,w=raw[0].shape[:2]
        assert h==w,(h,w)
        params=extract_camera_parameters(sim,self.names,h,w)
        inputs=dict(intrinsics=np.array([params[n]['intrinsics'] for n in self.names]),extrinsic_matrix=np.array([params[n]['extrinsic_matrix'] for n in self.names]))
        cid=sim.model.camera_name2id('agentview')
        poses,quats=u._pair_original_pose
        from scipy.spatial.transform import Rotation
        target=np.eye(4)
        target[:3,:3]=Rotation.from_quat(quats[cid][[1,2,3,0]]).as_matrix()
        target[:3,3]=poses[cid]
        targets=dict(intrinsics=np.array([params['agentview']['intrinsics']]),extrinsic_matrix=target[None])
        # Preserve policy noise sequence: image synthesis must not consume it.
        with torch.random.fork_rng(devices=[0]):
            pred=self.wrapper.infer([np.ascontiguousarray(x[::-1]) for x in raw],inputs,targets)
        assert pred.shape==(3,256,256) and np.isfinite(pred).all()
        restored=(pred.transpose(1,2,0)[::-1]*255).clip(0,255).astype(np.uint8)
        restored=np.asarray(Image.fromarray(restored).resize((w,h),Image.Resampling.BILINEAR)).copy()
        if suite not in self.saved:
            current_pos=sim.model.cam_pos[cid].copy(); current_quat=sim.model.cam_quat[cid].copy()
            before=sim.get_state().flatten().copy()
            try:
                sim.model.cam_pos[cid]=poses[cid]; sim.model.cam_quat[cid]=quats[cid]; sim.forward()
                gt=u._format_raw_obs(env._get_observations(force_update=True))['pixels'][keys[0]]
            finally:
                sim.model.cam_pos[cid]=current_pos; sim.model.cam_quat[cid]=current_quat; sim.forward()
                env._get_observations(force_update=True)
            assert np.array_equal(before,sim.get_state().flatten())
            # Upright display only; policy receives native raw convention.
            panel=np.concatenate([raw[0][::-1],restored[::-1],gt[::-1]],axis=1)
            Image.fromarray(panel).save(self.root/f'{suite}_shifted_synth_groundtruth.png')
            mse=float(np.mean((restored.astype(float)-gt.astype(float))**2))
            (self.root/f'{suite}_geometry.json').write_text(json.dumps(dict(inputs={k:v.tolist() for k,v in inputs.items()},targets={k:v.tolist() for k,v in targets.items()},psnr=float(10*np.log10(255**2/max(mse,1e-12)))),indent=2))
            self.saved.add(suite)
            print('NVS_PREFLIGHT',suite,'PSNR',10*np.log10(255**2/max(mse,1e-12)),flush=True)
        obs['pixels'][keys[0]]=restored[None]
        return obs
