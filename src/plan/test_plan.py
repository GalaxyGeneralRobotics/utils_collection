# example usage of plan.py

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

from src.mujoco.env_utils import add_state_to_config
from src.utils.data import get_traj_list, read_traj, get_traj
from src.utils.config import load_config
from src.utils.vis_plotly import Vis
from src.plan.plan import Planner

import numpy as np
from rich import print as rich_print

def get_dist(path):
    arm_path = path[:, :7]
    return np.abs(arm_path[1:]-arm_path[:-1]).sum()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/v3.1_norm')
    # parser.add_argument('--idx', type=str, default='0019/9032513d-27a9-4320-ae85-5d02a97e4791.npz') '0010/5d3186a2-a248-43be-91c2-56f013fd9fa7.npz'
    # parser.add_argument('--mode', type=str, default='0000')
    parser.add_argument('--vis', type=int, default=1)
    parser.add_argument('--idx', type=str, default=None)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--config', type=str, default='configs/mujoco.yaml')
    args = parser.parse_args()

    vis = Vis()
    traj_list = get_traj_list(args.root, args.mode)
    total = 0
    success = 0
    while True:
        if args.idx is None:
            idx = choice(traj_list)
            print(f'idx: {idx}')
        else:
            idx = args.idx
        traj = get_traj(idx, args.root, args.mode)
        init_state = read_traj(traj, 0, binary=1)
        last_state = read_traj(traj, len(traj['state'])-1, binary=1)
        last_state['qpos'][-2:] = 0

        config = load_config(args.config)['mujoco']
        config = add_state_to_config(config, init_state)

        planner = Planner(config, vis=args.vis)
        for _ in range(1):
            res, path = planner.plan(init_state['qpos'], last_state['qpos'], exec=args.vis)
            if res:
                path = np.array(path)
                rich_print(f'[bold blue]{_}: {get_dist(path)}[/bold blue]')
        planner.close()
        success += res
        total += 1
        rich_print(f'[bold green]success rate {success}/{total}[/bold green]')

if __name__ == '__main__':
    main()