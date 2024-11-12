# scripts/train_integrated.py

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from src.data import create_data_loaders

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Other imports
import yaml
import wandb
from tqdm import tqdm

# Local imports
from src.data.datasets import ChestXrayDataset
from src.models.integration import IntegratedModel
from src.utils.metrics import MetricTracker
from src.utils.checkpointing import CheckpointManager
from src.utils.optimization import CosineAnnealingWarmupRestarts
import traceback
import numpy as np


# Custom collate function for handling None values
def custom_collate(batch):
    """Custom collate function to handle None values in bounding boxes."""
    images, labels, bb_coords = zip(*batch)

    # Stack images and labels
    images = torch.stack(images)
    labels = torch.stack(labels)

    # Handle bounding boxes
    if any(bb is not None for bb in bb_coords):
        # Convert None to zero tensors
        bb_coords = [bb if bb is not None else torch.zeros(14, 4) for bb in bb_coords]
        bb_coords = torch.stack(bb_coords)
    else:
        bb_coords = None

    return images, labels, bb_coords


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train the integrated model.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training with fallback to single GPU."""
    try:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            gpu = int(os.environ['LOCAL_RANK'])

            torch.cuda.set_device(gpu)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            return rank, world_size
        else:
            print("Distributed environment variables not found.")
            print("Falling back to single GPU training.")
            if torch.cuda.is_available():
                torch.cuda.set_device(0)
            return 0, 1
    except Exception as e:
        print(f"Error setting up distributed training: {e}")
        print("Falling back to single GPU training.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1


def get_model(model):
    """Helper function to get model (handles both DDP and non-DDP cases)."""
    return model.module if isinstance(model, DDP) else model


def train_epoch(model, phase, train_loader, optimizer, scheduler, scaler, metric_tracker, is_main_process):
    """Train for one epoch."""
    model.train()
    metric_tracker.reset()

    if is_main_process:
        pbar = tqdm(total=len(train_loader), desc=f"Training ({phase['name']})")

    for batch_idx, (images, labels, bb_coords) in enumerate(train_loader):
        # Move data to GPU
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if bb_coords is not None:
            bb_coords = bb_coords.cuda(non_blocking=True)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, bb_coords, labels)
            loss = outputs['loss']

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        # Update metrics
        metric_tracker.update(
            outputs['logits'].detach(),
            labels,
            loss.item()
        )

        if is_main_process:
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.6f}"
            })

    if is_main_process:
        pbar.close()

    return metric_tracker.compute()


@torch.no_grad()
def validate(model, val_loader, metric_tracker, is_main_process):
    """Validate the model."""
    model.eval()
    metric_tracker.reset()

    if is_main_process:
        pbar = tqdm(total=len(val_loader), desc="Validation")

    for images, labels, bb_coords in val_loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        if bb_coords is not None:
            bb_coords = bb_coords.cuda(non_blocking=True)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, bb_coords, labels)

        metric_tracker.update(
            outputs['logits'].detach(),
            labels,
            outputs['loss'].item()
        )

        if is_main_process:
            pbar.update(1)

    if is_main_process:
        pbar.close()

    return metric_tracker.compute()


def main():
    # Parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup distributed training
    rank, world_size = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        print(f"Training with {world_size} GPU{'s' if world_size > 1 else ''}")

        # Create output directories
        os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['paths']['log_dir'], exist_ok=True)

        # Initialize wandb
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['run_name'],
            config=config
        )

    # In train_integrated.py

    # Create datasets
    train_dataset = ChestXrayDataset(
        image_dir=config['data']['train_image_dir'],
        label_file=config['data']['train_label_file'],
        bbox_file=config['data']['bbox_file'],
        transform=True
    )

    val_dataset = ChestXrayDataset(
        image_dir=config['data']['val_image_dir'],
        label_file=config['data']['val_label_file'],
        bbox_file=config['data']['bbox_file'],
        transform=False
    )

    test_dataset = ChestXrayDataset(
        image_dir=config['data']['test_image_dir'],
        label_file=config['data']['test_label_file'],
        bbox_file=config['data']['bbox_file'],
        transform=False
    )

    # Create data loaders
    loaders = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        distributed=(world_size > 1)
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # Create model
    model = IntegratedModel(
        pretrained_path=config['model']['vit_checkpoint'],
        num_classes=config['model']['num_classes'],
        freeze_vit=True,
        feature_dim=config['model']['feature_dim'],
        graph_hidden_dim=config['model']['graph_hidden_dim'],
        graph_num_heads=config['model']['graph_num_heads']
    ).cuda()

    # Wrap model with DDP if using distributed training
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True
        )

    # Initialize training components
    base_model = get_model(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['min_lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer=optimizer,
        first_cycle_steps=config['optimizer']['first_cycle_steps'],
        cycle_mult=1.0,
        max_lr=config['optimizer']['max_lr'],
        min_lr=config['optimizer']['min_lr'],
        warmup_steps=config['optimizer']['warmup_steps'],
        gamma=0.5
    )

    scaler = GradScaler()

    metric_tracker = MetricTracker(
        disease_names=[
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config['paths']['checkpoint_dir'],
        max_checkpoints=5,
        metric_name='mean_auc',
        mode='max'
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda')
        base_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resumed from checkpoint: {args.resume}")

    # Training phases
    phases = [
        {
            'name': 'graph_training',
            'epochs': config['training'].get('graph_epochs', 1),
            'unfreeze_layers': 0,
            'learning_rate': 1e-4
        },
        {
            'name': 'integration',
            'epochs': config['training'].get('integration_epochs', 1),
            'unfreeze_layers': 4,
            'learning_rate': 3e-4
        },
        {
            'name': 'fine_tuning',
            'epochs': config['training'].get('fine_tuning_epochs', 1),
            'unfreeze_layers': 4,
            'learning_rate': 5e-5
        }
    ]

    # Training loop
    total_epochs = sum(phase['epochs'] for phase in phases)
    current_epoch = start_epoch

    try:
        for phase in phases:
            if is_main_process:
                print(f"\nStarting {phase['name']} phase...")

            # Update model configuration for phase
            base_model.unfreeze_vit_layers(phase['unfreeze_layers'])

            # Update optimizer learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = phase['learning_rate']

            for epoch in range(phase['epochs']):
                current_epoch += 1

                if is_main_process:
                    print(f"\nEpoch {current_epoch}/{total_epochs}")

                # Set train sampler epoch
                if world_size > 1:
                    train_loader.sampler.set_epoch(current_epoch)

                # Train and validate
                train_metrics = train_epoch(model, phase, train_loader, optimizer,
                                            scheduler, scaler, metric_tracker, is_main_process)
                val_metrics = validate(model, val_loader, metric_tracker, is_main_process)

                # Log metrics and save checkpoint
                if is_main_process:
                    metrics = {
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                        'epoch': current_epoch,
                        'phase': phase['name']
                    }
                    wandb.log(metrics)

                    print("\nMetrics:")
                    print(f"Train Loss: {train_metrics['loss']:.4f}")
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val Mean AUC: {val_metrics['mean_auc']:.4f}")

                    # Save checkpoint
                    checkpoint_manager.save(
                        {
                            'epoch': current_epoch,
                            'model_state_dict': base_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'metrics': val_metrics,
                            'config': config
                        },
                        val_metrics['mean_auc'],
                        current_epoch
                    )

        # Final evaluation and model saving
        if is_main_process:
            # Load best model
            best_checkpoint = checkpoint_manager.load_best()
            base_model.load_state_dict(best_checkpoint['model_state_dict'])

            # Final validation
            final_metrics = validate(model, val_loader, metric_tracker, is_main_process)

            print("\nTraining completed!")
            print(f"Best validation Mean AUC: {final_metrics['mean_auc']:.4f}")

            # Save final model
            torch.save(
                {
                    'model_state_dict': base_model.state_dict(),
                    'final_metrics': final_metrics,
                    'config': config,
                    'timestamp': str(datetime.now())
                },
                Path(config['paths']['checkpoint_dir']) / 'final_model.pt'
            )

            wandb.finish()

    except Exception as e:
        print(f"Error during training: {e}")
        traceback.print_exc()
        if is_main_process and wandb.run is not None:
            wandb.finish()
        raise


if __name__ == '__main__':
    main()

