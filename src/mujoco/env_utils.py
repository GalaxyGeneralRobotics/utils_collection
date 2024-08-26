# example utils functions needed by env.py

import numpy as np
import torch
from transforms3d.quaternions import mat2quat, quat2mat

from src.utils.utils import to_numpy, to_torch
from src.utils.data import get_mesh_file, clip_point_cloud, get_urdf_file
from src.utils.config import DotDict
from src.utils.constants import IMAGE_INTRINSICS, IMAGE_EXTRINSICS, IMAGE_WIDTH, IMAGE_HEIGHT

def add_camera_to_config(config):
    rot = IMAGE_EXTRINSICS[:3, :3].copy()
    rot[:, 1:] *= -1
    config['camera'] = DotDict(
        intrinsics=IMAGE_INTRINSICS,
        pos=IMAGE_EXTRINSICS[:3, 3],
        quat=mat2quat(rot),
        w=IMAGE_WIDTH,
        h=IMAGE_HEIGHT,
        fovy=np.rad2deg(np.arctan(IMAGE_HEIGHT/2 / IMAGE_INTRINSICS[1, 1]) * 2),
    )
    return config

def add_table_to_config(config):
    obj = [DotDict(
        type='box',
        name='table',
        pos=np.array([0,0,-0.1]),
        quat=np.array([1.,0,0,0]),
        size=np.array([50,50,0.2]),
        fixed=True,
    )]
    config['obj'] = config.get('obj', []) + obj
    return config

def add_obj_to_config(config, obj_idx, obj_pose):
    obj = []
    for id, pose in zip(obj_idx, obj_pose):
        id = int(id)
        if id == -1:
            continue
        pose = to_numpy(pose)
        obj.append(DotDict(
            type='mesh',
            name=f'{id}',
            pos=pose[:3, 3],
            quat=mat2quat(pose[:3, :3]),
            path=get_mesh_file(id, mujoco=True),
            urdf=get_urdf_file(id),
            # path=[get_mesh_file(id)],
            fixed=False,
        ))
    config['obj'] = config.get('obj', []) + obj
    return config

def file_to_str(path):
    return path.replace('/', '_')

def extract_obj_pose(state, pad=None, batch=False):
    if batch:
        assert not pad is None
        results = [extract_obj_pose(s, pad) for s in state]
        return np.stack([x for x, y in results]), np.stack([y for x, y in results])
    obj_id = []
    new_obj_pose = []
    for name, pose in state['obj_pose'].items():
        if name == 'table':
            continue
        obj_id.append(int(name))
        new_obj_pose.append(pose)
    if pad is not None:
        left = pad - len(obj_id)
        assert left >= 0
        for _ in range(left):
            obj_id.append(-1)
            new_obj_pose.append(np.eye(4))
    return np.array(obj_id), np.stack(new_obj_pose)

def extract_pc(state, point_num=None, batch=False):
    if batch:
        assert not point_num is None
        return np.stack([extract_pc(s, point_num) for s in state])
    pc = clip_point_cloud(to_torch(state['pc'], torch.float32))
    if point_num is not None:
        if point_num < len(pc):
            idxs = np.random.choice(len(pc), (point_num,), replace=False)
        else:
            idxs = np.concatenate([np.arange(len(pc)), np.random.choice(len(pc), (point_num-len(pc),), replace=True)])
        pc = pc[idxs]
    return pc
 
def add_state_to_config(config, state):
    config = add_obj_to_config(config, state['obj_idx'], state['obj_pose'])
    config = add_table_to_config(config)
    config = add_camera_to_config(config)
    return config

def collect_state(state, key):
    return np.stack([s[key] for s in state])

def get_success(init_state, final_state):
    return (np.array([(fstate['obj_pose'][0, 2, 3] - istate['obj_pose'][0, 2, 3]) for istate, fstate in zip(init_state, final_state)]) > 0.05).astype(np.float32)