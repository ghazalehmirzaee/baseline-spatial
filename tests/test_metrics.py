# tests/test_metrics.py

import pytest
import torch
import numpy as np
from src.utils.metrics import MetricTracker


@pytest.fixture
def metric_tracker():
    disease_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    return MetricTracker(disease_names)


def test_metric_computation(metric_tracker):
    predictions = torch.rand(10, 14)
    labels = torch.randint(0, 2, (10, 14))
    loss = 0.5

    metric_tracker.update(predictions, labels, loss)
    metrics = metric_tracker.compute()

    assert 'mean_auc' in metrics
    assert 'mean_ap' in metrics
    assert 'mean_f1' in metrics
    assert 'loss' in metrics

