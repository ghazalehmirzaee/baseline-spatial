# scripts/train_integrated.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import argparse
from pathlib import Path
import yaml
import os
from tqdm import tqdm

from src.models.integration import IntegratedModel
from src.data.datasets import ChestXrayDataset
from src.utils.metrics import MetricTracker
from src.utils.checkpointing import CheckpointManager
from src.utils.optimization import CosineAnnealingWarmupRestarts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = -1
        gpu = -1

    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    return rank, world_size


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup distributed training
    rank, world_size = setup_distributed()
    is_main_process = rank == 0

    # Set up wandb
    if is_main_process:
        wandb.init(
            project=config['wandb']['project'],
            entity=config['wandb']['entity'],
            name=config['wandb']['run_name'],
            config=config
        )

    # Create datasets and dataloaders
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

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] // world_size,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Create model
    model = IntegratedModel(
        pretrained_path=config['model']['vit_checkpoint'],
        num_classes=config['model']['num_classes'],
        freeze_vit=True,
        feature_dim=config['model']['feature_dim'],
        graph_hidden_dim=config['model']['graph_hidden_dim'],
        graph_num_heads=config['model']['graph_num_heads']
    )

    # Move model to GPU and wrap with DDP
    model = model.cuda()
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config['optimizer']['first_cycle_steps'],
        cycle_mult=1.0,
        max_lr=config['optimizer']['max_lr'],
        min_lr=config['optimizer']['min_lr'],
        warmup_steps=config['optimizer']['warmup_steps'],
        gamma=0.5
    )

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()

    # Initialize metric tracker
    metric_tracker = MetricTracker(
        disease_names=[
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
    )# Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config['paths']['checkpoint_dir'],
        max_checkpoints=5,
        metric_name='mean_auc',
        mode='max'
    )

    # Training phases
    phases = [
        {
            'name': 'graph_training',
            # 'epochs': 10,
            'epochs': 1,
            'unfreeze_layers': 0,
            'learning_rate': 1e-4
        },
        {
            'name': 'integration',
            # 'epochs': 30,
            'epochs': 1,
            'unfreeze_layers': 4,  # Unfreeze last 4 ViT layers
            'learning_rate': 3e-4
        },
        {
            'name': 'fine_tuning',
            # 'epochs': 10,
            'epochs': 1,
            'unfreeze_layers': 4,
            'learning_rate': 5e-5
        }
    ]

    def train_epoch(phase: dict) -> dict:
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
            with autocast():
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

            # Gradient accumulation break
            if (batch_idx + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

        if is_main_process:
            pbar.close()

        metrics = metric_tracker.compute()
        return metrics

    @torch.no_grad()
    def validate() -> dict:
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

        metrics = metric_tracker.compute()
        return metrics

    # Training loop
    total_epochs = sum(phase['epochs'] for phase in phases)
    current_epoch = 0

    for phase in phases:
        if is_main_process:
            print(f"\nStarting {phase['name']} phase...")

        # Update model configuration for phase
        model.module.unfreeze_vit_layers(phase['unfreeze_layers'])

        # Update optimizer learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = phase['learning_rate']

        for epoch in range(phase['epochs']):
            current_epoch += 1

            if is_main_process:
                print(f"\nEpoch {current_epoch}/{total_epochs}")

            # Set train sampler epoch
            train_loader.sampler.set_epoch(current_epoch)

            # Train
            train_metrics = train_epoch(phase)

            # Validate
            val_metrics = validate()

            # Log metrics
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
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'metrics': val_metrics
                    },
                    val_metrics['mean_auc'],
                    current_epoch
                )

    # Final evaluation and model saving
    if is_main_process:
        # Load best model
        best_checkpoint = checkpoint_manager.load_best()
        model.module.load_state_dict(best_checkpoint['model_state_dict'])

        # Final validation
        final_metrics = validate()

        print("\nTraining completed!")
        print(f"Best validation Mean AUC: {final_metrics['mean_auc']:.4f}")

        # Save final models
        torch.save(
            {
                'model_state_dict': model.module.state_dict(),
                'final_metrics': final_metrics,
                'config': config
            },
            Path(config['paths']['checkpoint_dir']) / 'final_model.pt'
        )

        wandb.finish()

if __name__ == '__main__':
    main()

