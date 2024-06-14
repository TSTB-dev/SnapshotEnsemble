import os
import logging as log
log.basicConfig(level=log.INFO)
import argparse
import torch

from dataset import load_cifar10_dataloader, load_cifar100_dataloader, load_mnist_dataloader
from models import get_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="convnet")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_snapshots", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_single", action="store_true", default=False)
    return parser.parse_args()
    
def evaluate(args):
    device = torch.device(args.device)
    
    log.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, test_loader = load_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100_dataloader(args)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist_dataloader(args)
    else:
        raise ValueError("Invalid dataset")
    log.info("Dataset loaded")
    
    num_classes = 10 if args.dataset == "cifar10" or args.dataset == "mnist" else 100
    
    log.info(f"Loading model: {args.model}")
    snapshot_models = []
    for i in range(args.num_snapshots):
        model = get_model(args)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, f"snapshot_{i}.pth")))
        model.to(device)
        model.eval()
        snapshot_models.append(model)
    log.info("Model loaded")
    
    if args.eval_single:
        log.info("Evaluating Single model")
        for i, model in enumerate(snapshot_models):
            accuracy = inference_single(model, test_loader, device=device)
            log.info(f"Test Accuracy Single for Snapshot[{i}]: {accuracy}")
        log.info("Single model Evaluation complete")
    
    log.info("Evaluating Ensemble model")
    accuracy = inference_ensemble(snapshot_models, test_loader, device=device, num_classes=num_classes)
    log.info(f"Test Accuracy Ensemble: {accuracy}")
    

def inference_single(model, dataloader, device) -> float:
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return accuracy

def inference_ensemble(models, dataloader, device, num_classes) -> float:
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = torch.zeros(images.size(0), num_classes).to(device)
            for model in models:
                model.eval()
                outputs += model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    args = get_args()
    evaluate(args)
    
        
    
    
    
