import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import argparse
from pathlib import Path
from spikingjelly.activation_based import functional

from smodels import SEWResNet1D, PlainNet1D, SpikingResNet1D
from utils import (
    load_config,
    setup_logging,
    get_data_loaders,
    save_checkpoint,
    load_checkpoint
)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Reset neuron states after each batch
        functional.reset_net(model)
        
        total_loss = total_loss + loss.item()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # Convert percentage to actual number of correct predictions
        correct += (acc1.item() * target.size(0)) / 100.0
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # Log epoch-level metrics
    if wandb.run is not None:
        wandb.log({
            'train/epoch_loss': total_loss / len(train_loader),
            'train/epoch_accuracy': 100. * correct / total,
            'train/epoch': epoch,
            'train/learning_rate': optimizer.param_groups[0]['lr']
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, config):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            # Reset neuron states after each batch
            functional.reset_net(model)
            
            total_loss += loss.item()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # Convert percentage to actual number of correct predictions
            correct += (acc1.item() * target.size(0)) / 100.0
            total += target.size(0)
    
    # Log epoch-level validation metrics
    if wandb.run is not None:
        wandb.log({
            'val/epoch_loss': total_loss / len(val_loader),
            'val/epoch_accuracy': 100. * correct / total
        })
    
    return total_loss / len(val_loader), 100. * correct / total


def main(args):
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    
    # Setup device
    device = torch.device(config['training']['device'])
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            config=config,
            name=f"{config['model']['name']}_{config['model']['connect_f']}_T{config['training']['num_steps']}"
        )
    
    # Create model
    if config['model']['name'] == 'sew':
        model = SEWResNet1D(config['model']['connect_f'])
    elif config['model']['name'] == 'plain':
        model = PlainNet1D()
    elif config['model']['name'] == 'basic':
        model = SpikingResNet1D()
    else:
        raise NotImplementedError(f"Model {config['model']['name']} not implemented")
    
    model = model.to(device)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup tensorboard
    writer = SummaryWriter(config['logging']['log_dir'])
    
    # Training loop
    best_val_acc = 0
    start_epoch = 0
    
    if args.resume:
        start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, args.resume
        )
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, config)
        
        # Log metrics
        logging.info(
            f'Epoch {epoch}: '
            f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        )
        
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['logging']['save_dir'], 'best_model.pth')
            )
            
            # Log best model metrics
            if wandb.run is not None:
                wandb.log({
                    'best_val_accuracy': val_acc,
                    'best_val_loss': val_loss,
                    'best_epoch': epoch
                })
        
        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                os.path.join(config['logging']['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Test
    test_loss, test_acc = validate(model, test_loader, criterion, device, config)
    logging.info(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    # Log final test metrics
    if wandb.run is not None:
        wandb.log({
            'test/final_loss': test_loss,
            'test/final_accuracy': test_acc
        })
    
    writer.close()
    if wandb.run is not None:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable wandb logging')
    args = parser.parse_args()
    
    main(args) 