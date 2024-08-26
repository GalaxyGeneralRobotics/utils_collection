# example usage of env.py

import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(3):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

import argparse
from random import choice
from copy import deepcopy
from PIL import Image
import shutil

from src.mujoco.env import Env
from src.mujoco.env_utils import add_state_to_config
from src.utils.data import get_traj_list, read_traj, get_traj
from src.utils.config import load_config
from src.utils.ik import IK
from src.utils.vis_plotly import Vis

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/v4.1_norm_filter')
    # parser.add_argument('--idx', type=str, default='0019/9032513d-27a9-4320-ae85-5d02a97e4791.npz') '0010/5d3186a2-a248-43be-91c2-56f013fd9fa7.npz'
    # parser.add_argument('--mode', type=str, default='0000')
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--idx', type=str, default=None)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--config', type=str, default='configs/mujoco.yaml')
    args = parser.parse_args()

    traj_list = get_traj_list(args.root, args.mode)
    total = 0
    success = 0
    if args.vis and os.path.exists('tmp/vis'):
        shutil.rmtree('tmp/vis')
    while True:
        if args.idx is None:
            idx = choice(traj_list)
            print(f'idx: {idx}')
        else:
            idx = args.idx
        traj = get_traj(idx, args.root, args.mode)
        init_state = read_traj(traj, 0)
        frames = []

        config = load_config(args.config)['mujoco']
        config = add_state_to_config(config, init_state)
        env = Env(config)

        orig_state = env.reset(init_state['qpos'], require=dict(rgb=False, depth=False, pc=False))
        qpos = traj['joint_qpos']
        action = traj['action']
        for i in range(len(qpos)):
            q = read_traj(traj, i, binary=1)['action']
            state = env.act(q, require=dict(rgb=args.vis, depth=False, pc=False))
            # print(i)
            # if i + 17 < len(qpos):
                # print((q - state['qpos']) / (q - orig_state['qpos']))
            # print((q - state['qpos']))
            if args.vis:
                pixels = state['rgb']
                frames.append(pixels)
            # Image.fromarray(pixels).save('tmp/vis.png')
            orig_state = state
        if args.vis:
            os.makedirs('tmp/vis', exist_ok=True)
            Vis.save_video(np.stack(frames), f'tmp/vis/{idx.replace("/", "_")}.mp4', 20)

        start_pose = init_state['obj_pose'][init_state['obj_idx'].tolist().index(int(init_state['grasp_obj_id']))]
        final_pose = state['obj_pose'][str(int(init_state['grasp_obj_id']))]
        delta_height = final_pose[2, 3] - start_pose[2, 3]
        success += (delta_height > 0.05)
        total += 1
        print(f'{delta_height}, {success}/{total} {success/total}')
        # print(delta_height)
        # print(final_pose - start_pose)
        # print()

if __name__ == '__main__':
    main()