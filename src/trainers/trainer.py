# src/trainers/trainer.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import wandb
import os
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score


class Trainer:
    """
    Trainer class for Graph-Augmented ViT model.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optimizer: optim.Optimizer,
            scheduler: Optional[object] = None,
            device: str = 'cuda',
            wandb_config: Optional[Dict] = None,
            checkpoint_dir: str = 'checkpoints',
            num_epochs: int = 100,
            early_stopping_patience: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.patience = early_stopping_patience

        # Initialize mixed precision training
        self.scaler = GradScaler()

        # Setup directories
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize WandB
        if wandb_config:
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                name=wandb_config['run_name'],
                config=wandb_config
            )

        # Initialize best metrics
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'cls_loss': 0.0,
            'spatial_loss': 0.0,
            'anatomical_loss': 0.0
        }

        # Progress bar
        pbar = tqdm(total=len(self.train_loader), desc='Training')

        for batch in self.train_loader:
            images, labels, bb_coords = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            if bb_coords is not None:
                bb_coords = bb_coords.to(selfdevice)

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = self.model(images, bb_coords, labels=labels)
                loss = outputs['loss']

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['cls_loss'] += outputs['cls_loss'].mean().item()
            epoch_metrics['spatial_loss'] += outputs['spatial_loss']
            epoch_metrics['anatomical_loss'] += outputs['anatomical_loss']

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({k: f"{v/pbar.n:.4f}" for k, v in epoch_metrics.items()})

        pbar.close()

        # Average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)

        return epoch_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_labels = []
        total_preds = []
        val_loss = 0.0

        for batch in tqdm(self.val_loader, desc='Validation'):
            images, labels, bb_coords = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            if bb_coords is not None:
                bb_coords = bb_coords.to(self.device)

            outputs = self.model(images, bb_coords, labels=labels)
            val_loss += outputs['loss'].item()

            # Store predictions and labels
            preds = torch.sigmoid(outputs['logits'])
            total_preds.append(preds.cpu())
            total_labels.append(labels.cpu())

        # Concatenate all predictions and labels
        total_preds = torch.cat(total_preds, dim=0)
        total_labels = torch.cat(total_labels, dim=0)

        # Compute metrics
        metrics = self.compute_metrics(total_preds, total_labels)
        metrics['val_loss'] = val_loss / len(self.val_loader)

        return metrics

    def compute_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics."""

        metrics = {}

        # Compute per-class metrics
        for i in range(labels.shape[1]):
            class_preds = preds[:, i].numpy()
            class_labels = labels[:, i].numpy()

            # AUC-ROC
            if len(np.unique(class_labels)) > 1:
                metrics[f'class_{i}_auc'] = roc_auc_score(class_labels, class_preds)
            else:
                metrics[f'class_{i}_auc'] = 0.0

            # Average Precision
            metrics[f'class_{i}_ap'] = average_precision_score(class_labels, class_preds)

            # Convert predictions to binary
            binary_preds = (class_preds > 0.5).astype(int)

            # F1 Score
            metrics[f'class_{i}_f1'] = f1_score(class_labels, binary_preds)

            # Precision
            metrics[f'class_{i}_precision'] = precision_score(
                class_labels,
                binary_preds,
                zero_division=0
            )

            # Recall (Sensitivity)
            metrics[f'class_{i}_sensitivity'] = recall_score(
                class_labels,
                binary_preds,
                zero_division=0
            )

            # Specificity
            tn = np.sum((1 - class_labels) * (1 - binary_preds))
            fp = np.sum((1 - class_labels) * binary_preds)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'class_{i}_specificity'] = specificity

        # Compute mean metrics
        metrics['mean_auc'] = np.mean([
            metrics[k] for k in metrics if k.endswith('_auc')
        ])
        metrics['mean_ap'] = np.mean([
            metrics[k] for k in metrics if k.endswith('_ap')
        ])
        metrics['mean_f1'] = np.mean([
            metrics[k] for k in metrics if k.endswith('_f1')
        ])

        return metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'metrics': metrics,
            'config': self.model.get_state_dict()['config']
        }

        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}_auc_{metrics["mean_auc"]:.4f}.pt'
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model if applicable
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)

        # Save metrics to JSON
        metrics_path = os.path.join(
            self.checkpoint_dir,
            f'metrics_epoch_{epoch}.json'
        )
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def train(self) -> Dict[str, float]:
        """Main training loop."""
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['mean_auc'])

            # Log metrics
            metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            wandb.log(metrics, step=epoch)

            # Check for improvement
            current_auc = val_metrics['mean_auc']
            is_best = current_auc > self.best_val_auc

            if is_best:
                self.best_val_auc = current_auc
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Mean AUC: {val_metrics['mean_auc']:.4f}")
            print(f"Best Val Mean AUC: {self.best_val_auc:.4f} (Epoch {self.best_epoch + 1})")

        # Save final evaluation results
        final_results = {
            'best_epoch': self.best_epoch,
            'best_val_auc': self.best_val_auc,
            'final_metrics': val_metrics,
            'training_duration': str(datetime.now()),
        }

        results_path = os.path.join(self.checkpoint_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=4)

        return final_results

