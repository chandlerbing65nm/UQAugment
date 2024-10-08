# File: wandb_logging/metrics_logging.py

import wandb

def log_metrics(metrics_dict):
    wandb.log(metrics_dict)
