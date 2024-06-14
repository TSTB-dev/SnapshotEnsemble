import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import logging as log
log.basicConfig(level=log.INFO)
import argparse

from dataset import load_cifar10_dataloader, load_cifar100_dataloader, load_mnist_dataloader
from models import get_model
from lr import CyclicCosineLRScheduler

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="convnet")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_snapshots", type=int, default=5)
    parser.add_argument("--scheduler_type", type=str, default="cyclic_cosine")
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--cycle_length", type=int, default=-1)
    parser.add_argument("--cycle_length_decay", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--proj_name", type=str, default="snapshot_ensemble")
    
    args = parser.parse_args()
    return args

def train(args):
    log.info(f"Starting training with args: {args}")
    
    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    # wandb init
    wandb.init(project=args.proj_name)
    
    log.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, test_loader = load_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100_dataloader(args)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist_dataloader(args)
    else:
        raise ValueError("Invalid dataset")
    num_iters_per_epoch = len(train_loader)
    num_iters_total = args.num_epochs * num_iters_per_epoch
    log.info(f"Dataset loaded")
    
    log.info(f"Creating model: {args.model}")
    model = get_model(args).to(device)
    log.info(f"Model created")
    
    log.info(f"Creating optimizer: {args.optimizer}")
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.max_lr)
    else:
        raise ValueError("Invalid optimizer")
    
    if args.scheduler_type == "cyclic_cosine":
        cycle_length = num_iters_total // args.num_snapshots if args.cycle_length == -1 else args.cycle_length
        scheduler = CyclicCosineLRScheduler(optimizer, args.max_lr, args.min_lr, cycle_length, args.cycle_length_decay)
    else:
        raise ValueError("Invalid scheduler")
    
    cycle_count = 0
    log.info(f"Start training")
    for i in range(args.num_epochs):
        loss, acc, cycle_count = train_one_epoch(model, optimizer, scheduler, train_loader, device, \
            args.log_interval, i, args.num_epochs, num_iters_per_epoch, cycle_count)
    log.info(f"Training finished")

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: CyclicCosineLRScheduler,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    log_interval: int,
    epoch: int,
    num_epochs: int,
    num_iters_per_epoch: int,
    cycle_count: int = 0
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        optimizer.step()
        has_cycle_finished = scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if i % log_interval == 0:
            log.info(f"Epoch [{epoch}/{num_epochs}] Iter [{i}/{num_iters_per_epoch}] "
                     f"Loss: {total_loss / (i + 1):.4f} "
                     f"Acc: {correct / total:.4f} "
                     f"LR: {optimizer.param_groups[0]['lr']} "
                     f"Cycle: {cycle_count}")
            wandb.log({"train_loss": total_loss / (i + 1), "train_acc": correct / total})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})
        
        if has_cycle_finished:
            log.info(f"Cycle: {cycle_count} finished. Saving models...")
            # save wandb log dir.
            save_path = os.path.join(f"{wandb.run.dir}", f"snapshot_{cycle_count}.pth")
            torch.save(model.state_dict(), save_path)
            log.info(f"Models saved at model_{cycle_count}.pth")
            cycle_count += 1
    
    log.info(f"Epoch [{epoch}/{num_epochs}] "
             f"Loss: {total_loss / num_iters_per_epoch:.4f} "
             f"Acc: {correct / total:.4f}")
    
    return total_loss / num_iters_per_epoch, correct / total, cycle_count
    

if __name__ == "__main__":
    args = get_args()
    train(args)