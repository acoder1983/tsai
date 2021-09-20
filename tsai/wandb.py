# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/200_wandb.ipynb (unless otherwise specified).

__all__ = ['run_sweep']

# Cell
import os
from fastcore.script import *
from fastcore.xtras import *
from .imports import *
from .utils import *

# Cell
@call_parse
def run_sweep(
    sweep: Param("Path to YAML file with the sweep config", str),
    program: Param("Path to Python training script", str),
    launch: Param("Launch wanbd agent.", store_false) = True,
    count: Param("Number of runs to execute", int) = None,
    entity: Param("username or team name where you're sending runs", str) = None,
    project: Param("The name of the project where you're sending the new run.", str) = None,
    sweep_id: Param("Sweep ID. This option omits `sweep`", str) = None,
    relogin: Param("Relogin to wandb.", store_true) = False,
    login_key: Param("Login key for wandb", str) = None,
    tags: Param("Tag assigned to this run", str) = None,
):

    assert os.path.isfile(sweep), f"can't find file {sweep}"
    assert os.path.isfile(program), f"can't find file {program}"

    try:
        import wandb
    except ImportError:
        raise ImportError('You need to install wandb to run sweeps!')
    import yaml

    # Login to W&B
    if relogin:
        wandb.login(relogin=True)
    elif login_key:
        wandb.login(key=login_key)

    # Sweep id
    if not sweep_id:

        # Load the sweep config
        if isinstance(sweep, str): sweep = yaml2dict(sweep)

        # Initialize the sweep
        print('Initializing sweep...')
        sweep_id = wandb.sweep(sweep=sweep, entity=entity, project=project)
        print('...sweep initialized')

    # Load your training script
    print('Loading training script...')
    train_script, file_path = import_file_as_module(program, True)
    train_fn = getattr(train_script, "train")
    print('...training script loaded')


    # Launch agent
    if launch: print('\nRun additional sweep agents with:\n')
    else: print('\nRun sweep agent with:\n')
    print('    from a notebook:')
    print('        import wandb')
    print(f'        from {file_path} import train')
    print(f"        wandb.agent('{sweep_id}', function=train, count=None)\n")
    print('    from a terminal:')
    print(f"        wandb agent {os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{sweep_id}\n")
    if launch:
        print('Running agent...')
        wandb.agent(sweep_id, function=train_fn, count=count)