import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
from distutils.util import strtobool

import torch
import numpy as np
from tqdm import tqdm
from munch import Munch
from torch.utils import tensorboard
from torch.utils.data import DataLoader

from dataset import get_dataset
from models.fullmodel import FullModel

from train_utils import make_batch_for_epoch, test_single_task, train_single_task
from data_utils import get_dataset
from utils import set_random_seed, tprint
from stats import get_stats

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def validate(model, args, dataset, verbose=True):
    val_batch = make_batch_for_epoch(args, dataset)
    model.eval()

    accs = []

    for i, task_data in enumerate(zip(list(val_batch.values()))):
        acc = train_single_task(model, optimizer, task_data)
        accs.append(acc)
        
        if verbose:
            print(f"Val\t[{i+1}/{len(list(val_batch.values()))}]:{acc:.2f}%")
            
    
    return torch.mean(accs), torch.std(accs)
    


def train(args):
    train_set, val_set, _ = get_dataset(args)
    
    # classifier, model(embedding), 
    model = FullModel(args)
    device = 'cuda' if args.use_cuda else 'cpu'
    model.to(device)

    # optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.params, lr=args.lr)
        
        
    best_val_acc = -1.0
    best_epoch = 0
    count = 0
    
    for epoch in tqdm(range(args.n_epoch), desc="Overall training loop"):
        batch = make_batch_for_epoch(args, train_set) # support, support_gt, query, query_gt, src_stats, sup_stats = batch
        
        # iterate for all episodes(tasks)
        for i, task_data in enumerate(zip(list(batch.values()))):
            acc = train_single_task(model, optimizer, task_data)
            
            if (i+1) % 10 == 0:
                print(f"Train:[{i+1}/{len(list(batch.values()))}]:{acc:.2f}%")
                args.writer.add_scalar('train/acc', acc.item(), i+1)
            
                
        # validate
        val_acc, val_std = validate(model, args, val_set, verbose=False)
        print(f"Val Acc(Mean):{val_acc:.2f}\tVal Std:{val_std:.2f}")
        args.writer.add_scalar('val/acc', val_acc, epoch+1)
        args.writer.add_scalar('val/std', val_std, epoch+1)
        args.writer.flush() 
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_param = model.state_dict()
            best_epoch = epoch+1
            count = 0
        else:
            count += 1
            
        # early stopping -> threshold : 20
        if count == 20:
            break

    torch.save(best_param, os.path.join(args.log_path, f"model_{best_epoch}"))
    args.writer.close()
    print("Training has done!")
    
    

if __name__ == "__main__":
    # yaml + argparse together https://sungwookyoo.github.io/tips/ArgParser/
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="configuration file *.yml", type=str,  default='./configs/huffpost.yml')
    parser.add_argument('--args', help="whether to priortize args > yaml", type=bool,  default=False)
    args = parser.parse_args()

    # args priority is higher than yaml
    if not args.args:  
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
    # yaml priority is higher than args
    else:  
        opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(vars(args))

    # Logger
    opt['exp_name'] = tprint(opt['exp_name'])
    opt['log_path'] = os.path.join(opt['log_path'], opt['exp_name'])
    opt['log_path'].mkdir(parents=True, exist_ok=True)
    opt['writer'] = tensorboard.SummaryWriter(opt['log_path'])
    logger.info(f"SummaryWriter under {opt['log_path']}")

    print(json.dumps({k:str(v) for k, v in opt.items()}, indent=4))
    
    
    if opt['mode'] == 'train':
        train(Munch(opt))
    elif opt['mode'] == 'finetune':
        train(Munch(opt), finetune=True)
    else:
        raise ValueError
        
    # test(Munch(opt))
    
    