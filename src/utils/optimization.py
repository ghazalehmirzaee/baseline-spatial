# src/utils/optimization.py

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts and gradual learning rate warmup.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-5,
        warmup_steps: int = 0,
        gamma: float = 1.0
    ):
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        super().__init__(optimizer)

        self.init_lr()

    def init_lr(self):
        """Initialize learning rates for all param groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def get_lr(self) -> List[float]:
        """Calculate current learning rate."""
        if self.step_in_cycle < self.warmup_steps:
            # Warm up period
            return [
                self.min_lr + (base_max_lr - self.min_lr) *
                (self.step_in_cycle / self.warmup_steps)
                for base_max_lr in self.base_max_lrs
            ]

        # Cosine annealing
        progress = (self.step_in_cycle - self.warmup_steps) / (
            self.cur_cycle_steps - self.warmup_steps
        )
        return [
            self.min_lr + 0.5 * (base_max_lr - self.min_lr) *
            (1 + math.cos(math.pi * progress))
            for base_max_lr in self.base_max_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        """Update scheduler state and learning rate."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(
                    self.cur_cycle_steps * self.cycle_mult
                )
        else:
            self.cycle = epoch // self.first_cycle_steps
            self.step_in_cycle = epoch - self.cycle * self.first_cycle_steps
            self.cur_cycle_steps = self.first_cycle_steps * (
                self.cycle_mult ** self.cycle
            )

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

