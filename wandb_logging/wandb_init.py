# File: wandb_logging/wandb_init.py

import wandb

def initialize_wandb(args):
    wandb.init(
        project=args.wandb_project, 
        config=vars(args), 
        name=f'{args.model_name}',
        mode=args.wandb_mode,
    )
