"""Inference-only helpers extracted from the original pilot; no RL dependency."""
import hashlib
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, NEW_ROLLOUT_OPTION

BASE = None  # Set by evaluate.py from --parent.

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def success(info):
    if 'is_success' in info:
        return bool(np.asarray(info['is_success']).any())
    final = info.get('final_info', {})
    if isinstance(final, dict):
        return bool(np.asarray(final.get('is_success', False)).any())
    return any(isinstance(row, dict) and bool(row.get('is_success', False)) for row in final)
