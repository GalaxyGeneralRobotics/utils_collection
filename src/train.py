# example usage of config.py, logger.py

import os
import sys

ROOT_PATH = os.path.abspath(__file__)
for _ in range(2):
    ROOT_PATH = os.path.dirname(ROOT_PATH)
os.chdir(ROOT_PATH)
sys.path.append(ROOT_PATH)

import argparse
from datetime import datetime
from pprint import pprint
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.profiler import profile, record_function, ProfilerActivity
from math import sqrt

from src.network.model import get_model
from src.utils.data import get_mesh_name
from src.utils.dataset import CloseLoopDataset, Loader, get_sparse_tensor
from src.utils.logger import Logger
from src.utils.render import Renderer
from src.utils.config import load_config, add_argparse
from src.utils.utils import set_seed
from src.utils.constants import IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_INTRINSICS
from src.utils.vis_plotly import Vis

arg_mapping = [
    # (key in console, (key in config, type, default value))
    ('exp_name', ('exp_name', str, None)),
    ('type', ('model/type', str, None)), # key in config can be hierarchical
    ('batch_size', ('batch_size', int, None)), # key in config can be hierarchical
    ('backbone', ('model/backbone', str, None)),
    ('iter', ('max_iter', int, None)), 
    ('num_workers', ('num_workers', int, None)), 
    ('hist', ('common/len_history', int, None)), 
    ('pred', ('common/len_pred', int, None)), 
    ('est_type', ('common/est_type', str, None)), 
    ('delta_type', ('common/delta_type', str, None)), 
    ('input_width', ('model/input_width', int, None)), 
    ('delta_goal', ('model/delta_goal', str, None)), 
    ('delta_pc', ('model/delta_pc', int, None)), 
    ('no_input_joint', ('model/no_input_joint', int, None)), 
    ('diff_pred', ('model/diffusion/scheduler/prediction_type', str, None)), 
    ('yaml', ('yaml', str, os.path.join('configs', 'train.yaml'))), 
]

def main():
    # process config, seed, logger, device
    parser = argparse.ArgumentParser()
    add_argparse(parser, arg_mapping)
    args = parser.parse_args()
    config = load_config(args.yaml, arg_mapping, args)
    pprint(config)
    set_seed(config.seed)
    logger = Logger(config)
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    config.common.device = device

    # loading data, val loader can have multiple splits
    train_dataset = CloseLoopDataset(config.data, config.train_split)
    val_datasets = [CloseLoopDataset(config.data, split) for split in config.val_split]
    train_loader = Loader(DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, num_workers=config.num_workers, shuffle=True))
    val_loader = [Loader(DataLoader(dataset, batch_size=config.batch_size, drop_last=True, num_workers=config.num_workers, shuffle=True)) for dataset in val_datasets]

    # model and optimizer
    renderer = Renderer(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_INTRINSICS, 0)
    model = get_model(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, config.max_iter, eta_min=config.lr_min)

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        cur_iter = ckpt['iter']
        for _ in range(cur_iter):
            scheduler.step()
        print(f'loaded ckpt from {config.ckpt}')
    else:
        cur_iter = 0
    
    # training
    model.to(device)
    model.train()
    
    for it in trange(cur_iter, config.max_iter):
        optimizer.zero_grad()
        data = train_loader.get()
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,  with_stack=True) as prof:
        data = model.prepare_input(data, renderer, device, config.data.point_num, config.common.voxel_size)
        loss, result_dict = model(data)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
        # prof.export_chrome_trace(f"tmp/trace_{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        loss.backward()
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                if torch.isnan(p.grad).any():
                    print('nan')
                    p.grad.zero_()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        # print({k: v.mean().item() for k, v in result_dict.items()})
        if it % config.log_every == 0:
            logger.log({k: v.mean().item() for k, v in result_dict.items()}, 'train', it)
        
        if (it + 1) % config.save_every == 0:
            logger.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iter=it+1), it+1)

        if it % config.val_every == 0:
            with torch.no_grad():
                model.eval()
                for split, loader in zip(config.val_split, val_loader):
                    result_dicts = []
                    for _ in range(config.val_num):
                        data = loader.get()
                        data = model.prepare_input(data, renderer, device, config.data.point_num, config.common.voxel_size)
                        loss, result_dict = model(data)
                        result_dicts.append(result_dict)
                    logger.log({k: torch.cat([(dic[k] if len(dic[k].shape) else dic[k][None]) for dic in result_dicts]).mean() for k in result_dicts[0].keys()}, split, it)
                model.train()
    logger.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), iter=it+1), it+1)
    
if __name__ == '__main__':
    main()