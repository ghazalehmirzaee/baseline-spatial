# src/utils/optimization.py

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    Cosine annealing scheduler with warm restarts and gradual learning rate warmup.
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
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0

        # Initialize base_max_lrs
        self.base_max_lrs = [max_lr for _ in optimizer.param_groups]
        self.max_lrs = [max_lr for _ in optimizer.param_groups]  # Current max_lrs

        super().__init__(optimizer)

        # Initialize learning rates
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
                for base_max_lr in self.max_lrs
            ]

        # Cosine annealing
        progress = (self.step_in_cycle - self.warmup_steps) / (
                self.cur_cycle_steps - self.warmup_steps
        )
        if progress >= 1.0:
            progress = 1.0

        return [
            self.min_lr + 0.5 * (base_max_lr - self.min_lr) *
            (1 + math.cos(math.pi * progress))
            for base_max_lr in self.max_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        """Update scheduler state and learning rate."""
        if epoch is None:
            self.step_in_cycle = self.step_in_cycle + 1

            if self.step_in_cycle >= self.cur_cycle_steps:
                # Cycle completed, update cycle parameters
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(
                    self.cur_cycle_steps * self.cycle_mult
                )
                # Update max_lrs for the new cycle
                self.max_lrs = [
                    base_max_lr * (self.gamma ** self.cycle)
                    for base_max_lr in self.base_max_lrs
                ]
        else:
            self.cycle = epoch // self.first_cycle_steps
            self.step_in_cycle = epoch - self.cycle * self.first_cycle_steps
            self.cur_cycle_steps = self.first_cycle_steps * (
                    self.cycle_mult ** self.cycle
            )
            # Update max_lrs based on current cycle
            self.max_lrs = [
                base_max_lr * (self.gamma ** self.cycle)
                for base_max_lr in self.base_max_lrs
            ]

        self.last_epoch = math.floor(epoch) if epoch is not None else self.last_epoch + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr if hasattr(self, '_last_lr') else self.get_lr()

