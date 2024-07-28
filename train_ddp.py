# python -m torch.distributed.launch train_ddp.py  --multiprocessing-distributed --exp_name 0710_test_new --model_size b

from __future__ import print_function
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
import builtins
from tqdm import tqdm
import os

from engine import train, validate

from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from robust_segment_anything.utils.transforms import ResizeLongestSide 

from dataset import TrainDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--num_points', type=int, default=3, help='Number of point prompts used during training')

parser.add_argument("--continue_training", action='store_true')
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--model_size", type=str, default="l")
parser.add_argument("--data_dir", type=str, default="data/all_data")
parser.add_argument("--save_dir", type=str, default="checkpoints")


parser.add_argument(
    "--world-size",
    default=1,
    type=int,
    help="number of nodes for distributed training",
)

parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)

parser.add_argument(
    "--local_rank", default=0, type=int, help="node rank for distributed training"
)

parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)

parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

opt = parser.parse_args()

def main(opt):   
    if opt.exp_name == "":
        print('Please enter the experiment name!!!')
        # breakpoint()

    if opt.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):  

    opt.gpu = gpu

    # suppress printing if not master
    if opt.multiprocessing_distributed and opt.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    # if opt.gpu is not None:
    #     print("Use GPU: {} for training".format(opt.gpu))        

    if opt.distributed:
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.local_rank = opt.local_rank * ngpus_per_node + gpu
            
        dist.init_process_group(
            backend=opt.dist_backend,
            world_size=opt.world_size,
            rank=opt.local_rank,
        )    
        
    train_flag = False
    
    if opt.continue_training:
        model_sam_path = '{}/{}_best.pth'.format(opt.save_dir, opt.exp_name)        
        train_flag = True         
        print('Using pretrained checkpoint. Model Path: {} ...'.format(model_sam_path))
    
    else: 
        model_sam_path = None
        print('Train from scratch ... ')
        
    model = sam_model_registry["vit_{}".format(opt.model_size)](opt=opt, checkpoint=model_sam_path, train=train_flag)    

    if opt.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        find_unused_parameters = True
        if opt.gpu is not None:
            torch.cuda.set_device(opt.gpu)
            model.cuda(opt.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opt.workers = int((opt.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[opt.gpu], find_unused_parameters=find_unused_parameters
            )

        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[opt.gpu]
            )
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
           
    elif opt.gpu is not None:
        torch.cuda.set_device(opt.gpu)
        model.cuda(opt.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    else:
        model.cuda()
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")              
   
    train_set = TrainDataset(opt=opt, mode='train')
    val_set = TrainDataset(opt=opt, mode='val')
    
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=(train_sampler is None),        
                              num_workers=opt.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=(val_sampler is None),
                            num_workers=opt.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 10)
    best_loss = 9999.
    
    try:
        sam_transform = ResizeLongestSide(model.module.image_encoder.img_size)
        
    except: 
        sam_transform = ResizeLongestSide(model.image_encoder.img_size)

    print(opt)
    print('====================Start training======================')
    print('Model checkpoint will be saved to {} ... '.format(opt.save_dir))
    os.makedirs(opt.save_dir, exist_ok=True)
    
    for epoch in range(1, opt.epochs + 1):  
        print('Epoch {} ...'.format(epoch))  
            
        print('==> Training')
        train(opt, epoch, optimizer, train_loader, sam_transform, model) 

        if opt.gpu == 0:
            print('==> Validation')
            val_loss = validate(opt, epoch, val_loader, sam_transform, model)
        
            print('Validation loss:{}'.format(val_loss))
            print('Best val loss so far: {}'.format(best_loss))
    
            if val_loss < best_loss:
                best_loss = val_loss
                print('Loss is lowest in epoch {}'.format(epoch))
                model_saved_path = '{}/{}_best.pth'.format(opt.save_dir, opt.exp_name)
                torch.save(model.state_dict(), model_saved_path)  
                print('Current best model checkpoint saved to {}'.format(model_saved_path)) 
    
    if opt.gpu == 0:          
        print('Training complete! Saving last model!')
        model_saved_path = '{}/{}_last.pth'.format(opt.save_dir, opt.exp_name)
        torch.save(model.state_dict(), model_saved_path) 
        print('Last model saved to {}'.format(model_saved_path)) 

if __name__ == "__main__":
    main(opt)
