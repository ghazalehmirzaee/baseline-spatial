# scripts/train.py

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml

from src.models.integration import GraphAugmentedViT
from src.trainers.trainer import Trainer
from src.data.datasets import ChestXrayDataset
from src.utils.optimization import CosineAnnealingWarmupRestarts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for training')
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Create model
    model = GraphAugmentedViT(
        num_classes=config['model']['num_classes'],
        vit_pretrained=config['model']['vit_pretrained'],
        vit_checkpoint=config['model']['vit_checkpoint'],
        feature_dim=config['model']['feature_dim'],
        graph_hidden_dim=config['model']['graph_hidden_dim'],
        graph_num_heads=config['model']['graph_num_heads'],
        graph_dropout=config['model']['graph_dropout'],
        fusion_type=config['model']['fusion_type']
    )

    # Move model to device
    model = model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['lr'],
        weight_decay=config['optimizer']['weight_decay']
    )

    # Create scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=config['optimizer']['first_cycle_steps'],
        cycle_mult=config['optimizer']['cycle_mult'],
        max_lr=config['optimizer']['max_lr'],
        min_lr=config['optimizer']['min_lr'],
        warmup_steps=config['optimizer']['warmup_steps'],
        gamma=config['optimizer']['gamma']
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        wandb_config=config['wandb'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        num_epochs=config['training']['num_epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    # Train model
    results = trainer.train()
    print("\nTraining completed!")
    print(f"Best validation AUC: {results['best_val_auc']:.4f}")
    print(f"Best epoch: {results['best_epoch'] + 1}")


if __name__ == '__main__':
    main()

