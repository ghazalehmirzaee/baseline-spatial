
# src/utils/logging.py

import wandb
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

class Logger:
    """Unified logging interface for experiments."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        wandb_config: Optional[Dict] = None
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up file logging
        self.setup_file_logging()

        # Initialize W&B if config provided
        self.use_wandb = wandb_config is not None
        if self.use_wandb:
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config['entity'],
                name=experiment_name,
                config=wandb_config
            )

    def setup_file_logging(self) -> None:
        """Set up file logging handler."""
        log_file = self.log_dir / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """Log metrics to all configured outputs."""
        # Log to file
        logging.info(f"Step {step}: {metrics}")

        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)

        # Save metrics to JSON
        metrics_file = self.log_dir / f"metrics_{self.experiment_name}.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        if step is not None:
            all_metrics[str(step)] = metrics
        else:
            all_metrics['latest'] = metrics

        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=4)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        # Log to file
        logging.info(f"Hyperparameters: {params}")

        # Log to W&B
        if self.use_wandb:
            wandb.config.update(params)

        # Save to JSON
        params_file = self.log_dir / f"params_{self.experiment_name}.json"
        with open(params_file, 'w') as f:
            json.dump(params, f, indent=4)

    def log_model_summary(self, model: torch.nn.Module) -> None:
        """Log model architecture summary."""
        # Get model summary
        summary = str(model)

        # Log to file
        logging.info(f"Model Architecture:\n{summary}")

        # Save to text file
        summary_file = self.log_dir / f"model_summary_{self.experiment_name}.txt"
        with open(summary_file, 'w') as f:
            f.write(summary)

    def finish(self) -> None:
        """Clean up logging."""
        if self.use_wandb:
            wandb.finish()

