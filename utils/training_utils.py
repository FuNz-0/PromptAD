import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(TASK, root_dir, **kwargs):

    k_shot = kwargs['k_shot']
    dataset = kwargs['dataset']

    csv_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'csv')
    check_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'checkpoint')
    img_dir = os.path.join(root_dir, f'{dataset}', f'k_{k_shot}', 'imgs')

    csv_path = os.path.join(csv_dir, f"Seed_{kwargs['seed']}-results.csv")
    check_path = os.path.join(check_dir, f"{TASK}-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt")

    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return img_dir, csv_path, check_path
