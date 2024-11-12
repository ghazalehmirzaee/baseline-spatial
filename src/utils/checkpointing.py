# src/utils/checkpointing.py

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime

class CheckpointManager:
    """Manage model checkpoints and logging."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        metric_name: str = 'mean_auc',
        mode: str = 'max'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode

        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.checkpoints = []

    def is_better(self, metric: float) -> bool:
        """Check if current metric is better than best so far."""
        if self.mode == 'max':
            return metric > self.best_metric
        return metric < self.best_metric

    def save(
        self,
        state: Dict[str, Any],
        metric: float,
        epoch: int
    ) -> None:
        """Save checkpoint and manage checkpoint history."""
        # Create checkpoint filename
        filename = (
            f"checkpoint_epoch_{epoch}_{self.metric_name}_{metric:.4f}.pt"
        )
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(state, filepath)

        # Update checkpoint history
        self.checkpoints.append({
            'path': filepath,
            'metric': metric,
            'epoch': epoch
        })

        # Sort checkpoints by metric
        self.checkpoints.sort(
            key=lambda x: x['metric'],
            reverse=(self.mode == 'max')
        )

        # Remove old checkpoints if exceeding max_checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            checkpoint_to_remove = self.checkpoints.pop()
            if checkpoint_to_remove['path'].exists():
                checkpoint_to_remove['path'].unlink()

        # Update best metric
        if self.is_better(metric):
            self.best_metric = metric
            # Save best model separately
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(state, best_path)

    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the latest checkpoint."""
        if not self.checkpoints:
            return None

        latest_checkpoint = max(self.checkpoints, key=lambda x: x['epoch'])
        return torch.load(latest_checkpoint['path'])

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            return torch.load(best_path)
        return None


