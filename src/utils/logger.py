# example can be found in src/train.py

import os
from os.path import dirname, join, abspath
import glob
from datetime import datetime
import yaml
import torch
import wandb
import subprocess
import shutil
from rich import print as rich_print
from pygments import highlight
from pygments.lexers import DiffLexer, PythonLexer
from pygments.formatters import HtmlFormatter

from src.utils.config import to_dict

def get_git_status(command):
    try:
        git_command = dict(
            status=['git', 'status'],
            diff=['git', 'diff'],
            diff_staged=['git', 'diff', '--staged'],
            id=['git', 'rev-parse', 'HEAD']
        )[command]
        result = subprocess.check_output(git_command)
        return result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error getting current commit ID: {e}")
        return None

def save_diff_with_syntax_highlighting(diff_text, output_file):
    lexer = DiffLexer()
    formatter = HtmlFormatter(full=True, linenos=True)
    highlighted_diff = highlight(diff_text, lexer, formatter)
    
    with open(output_file, 'w') as f:
        f.write(highlighted_diff)
        
def save_all_python_files(directory, output_file):
    # remove the output_file directory
    if os.path.exists(output_file):
        shutil.rmtree(output_file)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # copy file
                target_path = join(output_file, file_path[len('./src/'):])
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(file_path, target_path)
    # print to console in hightlight to tell me the code has been saved
    rich_print('[bold green]All python files have been saved[/bold green]')

class Logger:
    def __init__(self, config: dict):
        """
            automatically create experiment directory and save config
            config needs to have at least 'exp_name'
        """
        self.config = config
        if config.exp_name != 'temp':
            wandb.require("core")
            wandb.init(project='close_loop', 
                    name=config.exp_name, 
                    config=config,
                    settings=wandb.Settings(_disable_stats=True, _disable_meta=True))
            wandb.run.log_code(root='./src')

        # create exp directory
        exp_path = join('exps', config.exp_name)
        os.makedirs(exp_path, exist_ok=config.exp_name == 'temp') # other experiments should not overwrite
        log_path = join(exp_path, 'log')
        os.makedirs(log_path, exist_ok=True)
        self.ckpt_path = join(exp_path, 'ckpt')
        os.makedirs(self.ckpt_path, exist_ok=True)

        # save config
        with open(join(exp_path, 'config.yaml'), 'w') as f:
            yaml.dump(to_dict(config), f)
        with open(join(exp_path, 'git_status.txt'), 'w') as f:
            f.write(f"Commit ID: {get_git_status('id')}\n")
            f.write('\n')
            f.write('\n')
            f.write(get_git_status('status'))
            f.write('\n')
            f.write('\n')
            f.write(get_git_status('diff'))
            f.write('\n')
            f.write('\n')
            f.write(get_git_status('diff_staged'))
        save_diff_with_syntax_highlighting(get_git_status('diff'), join(exp_path, 'diff.html'))
        save_diff_with_syntax_highlighting(get_git_status('diff_staged'), join(exp_path, 'diff_staged.html'))
        save_all_python_files('./src', join(exp_path, 'code'))

    def log(self, dic: dict, mode: str, step: int):
        """
            log a dictionary, requires all values to be scalar
            mode is used to distinguish train, val, ...
            step is the iteration number
        """
        if self.config.exp_name != 'temp':
            wandb.log({f'{mode}/{k}': v for k, v in dic.items()}, step=step)
    
    def save(self, dic: dict, step: int):
        """
            save a dictionary to a file
        """
        torch.save(dic, join(self.ckpt_path, f'ckpt_{step}.pth'))