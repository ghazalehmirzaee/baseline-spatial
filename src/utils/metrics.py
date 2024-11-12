# src/utils/metrics.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple


class MetricTracker:
    """Track and compute various metrics during training."""

    def __init__(self, disease_names: List[str]):
        self.disease_names = disease_names
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.labels = []
        self.losses = []

    def update(
            self,
            preds: torch.Tensor,
            labels: torch.Tensor,
            loss: float
    ) -> None:
        """Update metrics with batch results."""
        self.predictions.append(preds.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        predictions = torch.cat(self.predictions, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()

        metrics = {}

        # Compute per-disease metrics
        for i, disease in enumerate(self.disease_names):
            # AUC-ROC
            if len(np.unique(labels[:, i])) > 1:
                metrics[f"{disease}_auc"] = roc_auc_score(
                    labels[:, i],
                    predictions[:, i]
                )
            else:
                metrics[f"{disease}_auc"] = 0.0

            # Average Precision
            metrics[f"{disease}_ap"] = average_precision_score(
                labels[:, i],
                predictions[:, i]
            )

            # Other metrics for binary predictions
            binary_preds = (predictions[:, i] > 0.5).astype(int)

            # F1 Score
            tp = np.sum((labels[:, i] == 1) & (binary_preds == 1))
            fp = np.sum((labels[:, i] == 0) & (binary_preds == 1))
            fn = np.sum((labels[:, i] == 1) & (binary_preds == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            metrics[f"{disease}_precision"] = precision
            metrics[f"{disease}_recall"] = recall
            metrics[f"{disease}_f1"] = f1

        # Compute mean metrics
        metrics["mean_auc"] = np.mean([
            metrics[k] for k in metrics if k.endswith("_auc")
        ])
        metrics["mean_ap"] = np.mean([
            metrics[k] for k in metrics if k.endswith("_ap")
        ])
        metrics["mean_f1"] = np.mean([
            metrics[k] for k in metrics if k.enmetrics[k] for k in metrics if k.endswith("_f1")
        ])

        # Add loss
        metrics["loss"] = np.mean(self.losses)

        return metrics

